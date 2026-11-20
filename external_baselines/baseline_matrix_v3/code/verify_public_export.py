"""Cold verify a sanitized public namespace and optionally recompute its scores.

Original hash fields describe original source bytes. PUBLIC_CONTENT_REFS binds
those identities to the derived public files and their normalized Git bodies.
This verifier never contacts a server or requires model/image assets.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True


def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(condition, description):
    if not condition: raise ValueError(description)


def evidence_path(value):
    path = str(value).replace('\\', '/')
    for marker in ('/data/registry/v3/', '/data/refs/v3/'):
        if marker in path: return 'configs/datasets/v3/' + path.split(marker, 1)[1]
    for marker in ('/results/', '/configs/'):
        if marker in path: return marker.strip('/') + '/' + path.split(marker, 1)[1]
    raise ValueError('Cannot resolve evidence template: ' + path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(root, archive=None, recompute_scores=False):
    root = Path(root).resolve(); index_path = root / 'PUBLIC_CONTENT_REFS.json'; index = read(index_path)
    require(index['schema'] == 'baseline_matrix.public_content_refs.v1', 'Unknown public index schema')
    entries = index['files']; total = 0; transformed = 0; jsonl_lines = 0
    for relative, info in entries.items():
        path = root / relative
        require(path.resolve().is_relative_to(root), 'Indexed path escapes namespace')
        body = path.read_bytes()
        require(len(body) == info['public_bytes'] and sha(body) == info['public_sha256'], 'Public body identity mismatch: ' + relative)
        lf = body if path.suffix == '.npy' else body.replace(b'\r\n', b'\n')
        oid = hashlib.sha1(f'blob {len(lf)}\0'.encode() + lf).hexdigest()
        require(sha(lf) == info['git_lf']['sha256'] and oid == info['git_lf']['blob_oid_sha1'], 'Git LF identity mismatch: ' + relative)
        require(len(lf) == info['git_lf']['bytes'], 'Git LF length mismatch')
        if info['transformation'] == 'EXACT': require(info['source_sha256'] == info['public_sha256'], 'False exact source assertion')
        if 'line_identities' in info:
            lines = body.splitlines(keepends=True)
            for line in info['line_identities']:
                actual = lines[line['line'] - 1]
                require(sha(actual) == line['public_sha256'], 'Public JSONL line identity mismatch')
                require(json.loads(actual).get('id') == line['id'], 'Public JSONL line ID mismatch')
                jsonl_lines += 1
        total += len(body); transformed += info['transformation'] == 'DERIVED_PATH_TEMPLATES'
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    require(actual == set(entries) | {'PUBLIC_CONTENT_REFS.json'}, 'Unindexed or missing public files')
    if archive:
        prefix = index['namespace'] + '/'
        with zipfile.ZipFile(archive) as zipped:
            expected = {prefix + path for path in entries} | {prefix + 'PUBLIC_CONTENT_REFS.json'}
            require(set(zipped.namelist()) == expected, 'Archive member set mismatch')
            require(zipped.read(prefix + 'PUBLIC_CONTENT_REFS.json') == index_path.read_bytes(), 'Archive index differs')
            for relative, info in entries.items():
                require(sha(zipped.read(prefix + relative)) == info['public_sha256'], 'Archive payload differs: ' + relative)
    raw_runs = 0
    for relative in entries:
        if not relative.startswith('results/') or not relative.endswith('/complete.json'): continue
        complete = read(root / relative)
        if not all(key in complete for key in ('manifest_sha256', 'records_jsonl_sha256')): continue
        folder = relative.rsplit('/', 1)[0]
        for name, key in [('manifest.json', 'manifest_sha256'), ('records.jsonl', 'records_jsonl_sha256')]:
            target = folder + '/' + name
            require(target in entries and entries[target]['source_sha256'] == complete[key], 'Original raw SHA graph is broken: ' + target)
        manifest = read(root / folder / 'manifest.json'); rows = jsonl(root / folder / 'records.jsonl')
        require(len(rows) == complete['expected_QA'] == complete['recorded_QA'], 'Raw run coverage mismatch')
        require([row['id'] for row in rows] == manifest['expected_ids'], 'Raw run expected order differs')
        require(len({row['id'] for row in rows}) == len(rows), 'Duplicate raw QA')
        require(dict(Counter(row['status'] for row in rows)) == complete['counts'], 'Raw status counts differ')
        raw_runs += 1
    scoring = []
    if recompute_scores:
        sys.path.insert(0, str(root / 'code'))
        from metrics import score_question
        import numpy as np
        for relative in entries:
            if not relative.startswith('results/scored/') or not relative.endswith('/summary.json'): continue
            summary = read(root / relative); folder = relative.rsplit('/', 1)[0]
            scored_path = folder + '/scored_records.jsonl'
            if scored_path not in entries: continue
            scored = jsonl(root / scored_path)
            require(len(scored) == summary['expected_run_questions'], 'Scored coverage differs')
            for name, expected in summary['derived_file_hashes'].items():
                require(entries[folder + '/' + name]['source_sha256'] == expected, 'Original scoring SHA graph broken')
            raw_folder = evidence_path(summary['integrity']['raw_dir'])
            manifest = read(root / raw_folder / 'manifest.json')
            raw_rows = jsonl(root / raw_folder / 'records.jsonl')
            raw_identity = {item['line']: item for item in entries[raw_folder + '/records.jsonl']['line_identities']}
            raw_lookup = {row['id']: row for row in raw_rows}
            inputs = evidence_path(manifest['config']['inputs'])
            refs_path = inputs.replace('_inputs.jsonl', '_refs.jsonl')
            refs = {row['id']: row for row in jsonl(root / refs_path)}
            require(set(raw_lookup) == {row['id'] for row in scored}, 'Scored/raw ID cohorts differ')
            all_pass = True
            for row in scored:
                original = raw_lookup[row['id']]
                require(row['raw_status'] == original['status'], 'Scored status differs from raw')
                identity = raw_identity[row['raw_jsonl_line_number']]
                require(identity['source_sha256'] == row['raw_jsonl_exact_line_sha256'] and identity['id'] == row['id'], 'Scored/raw original line binding differs')
                if row['raw_status'] != 'PASS':
                    require(row['score'] is None, 'Failure assigned a fabricated score'); all_pass = False; continue
                for key in ('raw_prediction', 'output_token_ids', 'original_image_sha256', 'actual_visual_token_count', 'stored_visual_KV_bytes'):
                    require(row.get(key) == original.get(key), 'Scored content changed from raw: ' + key)
                metric = score_question(row['dataset'], original['raw_prediction'], refs[row['id']])
                require(abs(metric['score'] - row['score']) < 1e-12 and metric['metric_id'] == row['metric_id'], 'Score does not recompute')
                require(metric['reference_texts'] == row['reference_texts'], 'Scored references differ from frozen references')
            if all_pass:
                mean = sum(row['score'] for row in scored) / len(scored)
                require(abs(mean - summary['primary_mean']) < 1e-12, 'Reported mean differs from recomputed mean')
                clusters = defaultdict(list)
                for row in scored: clusters[row['source_cluster_id']].append(row['score'])
                ordered = sorted(clusters)
                sums = np.asarray([sum(clusters[key]) for key in ordered], dtype=np.float64)
                sizes = np.asarray([len(clusters[key]) for key in ordered], dtype=np.float64)
                if len(ordered) >= 2:
                    bootstrap = summary['bootstrap']; gen = np.random.default_rng(bootstrap['seed'])
                    draws = gen.integers(0, len(ordered), size=(bootstrap['repetitions'], len(ordered)))
                    interval = np.quantile(sums[draws].sum(1) / sizes[draws].sum(1), [.025, .975]).tolist()
                    require(np.allclose(interval, summary['primary_ci95'], rtol=0, atol=1e-12), 'Reported cluster CI differs')
            else: require(summary['primary_mean'] is None, 'Failed run reported a full-cohort mean')
            scoring.append({'summary': relative, 'questions': len(scored), 'all_pass': all_pass,
                            'mean': summary['primary_mean'], 'metric_ids': summary['metric_ids']})
    return {'status': 'PASS_PUBLIC_EXPORT_VERIFIED', 'public_files': len(entries), 'public_payload_bytes': total,
            'derived_path_files': transformed, 'jsonl_line_identities': jsonl_lines, 'raw_complete_runs_verified': raw_runs,
            'score_recomputations': scoring, 'archive_verified': bool(archive),
            'original_images_weights_or_server_access_required': False, 'github_upload_verified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--recompute-scores', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    value = verify(args.root, args.archive, args.recompute_scores)
    if args.receipt:
        with args.receipt.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == '__main__': main()
