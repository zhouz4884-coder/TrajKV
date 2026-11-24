"""Prepare an immutable, inspected public increment. Never contacts GitHub.

All large output is confined to the named E evidence volume. The original
project, result mirrors and earlier releases are read-only inputs. Public raw
files are explicitly derived: path fields are templated, while original SHA
fields retain their source identity and are resolved by PUBLIC_CONTENT_REFS.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('E:/baseline_matrix_v3')
NAMESPACE = 'external_baselines/baseline_matrix_v3'
TEXT_SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.toml', '.yaml', '.yml', '.cu', '.cpp', '.cuh', '.h', '.sh', '.md', '.txt', '.jinja', '.cfg', '.ini'}
BLOCK_DIRS = {'.git', '__pycache__', '.venv', 'node_modules', '.ipynb_checkpoints'}
DATA_FIELDS = {'question', 'raw_question', 'rendered_question', 'raw_prediction', 'prediction', 'raw_decode_with_special_tokens',
               'reference_texts', 'answers', 'answer', 'normalized', 'normalized_prediction', 'normalized_references', 'full_prompt'}
PRIVATE_DOC_PREFIXES = ('EXECUTION_', 'RECOVERY_', 'CURRENT_')
AUDIT_ROOT_GLOBS = ('*acceptance*.json', '*support*.json', '*gate*.json', '*control*.json', '*diagnostic_status*.json',
                    'M1_official_identity_verification_v2.json', 'M1_pinned_official_tree_v1.json',
                    'model_final_inventory_v1.json', 'model_artifact_remote_rehash_v1.json', 'paired_*.json',
                    '*identity*equivalence*.json', '*source_proof*.json')
AUDIT_DIRS = ('upstream', 'source_snapshots', 'dataset_sources', 'dataset_staging', 'support_status',
              'metrics_gate_v2_fixtures', 'transformers_5_12_0', 'scorer_sources')
RESULT_DIRS = ('native_p0', 'qwen_bridge', 'p0')
P1_OUT_OF_SCOPE_SOURCE = {'code/methods/turboquant_buffered_adapter.py', 'code/gates/turboquant_m1_diagnostic.py'}


def digest(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git_identity(raw):
    return {'sha256': digest(raw), 'blob_oid_sha1': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest(), 'bytes': len(raw)}
def json_bytes(value): return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def inside(path, root):
    try: path.resolve().relative_to(root.resolve()); return True
    except ValueError: return False


def source_files(folder):
    if not folder.exists(): return
    for path in folder.rglob('*'):
        if not path.is_file() or set(path.relative_to(folder).parts) & BLOCK_DIRS: continue
        if path.suffix.lower() not in TEXT_SUFFIXES | {'.npy'} and not path.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')): continue
        if path.stat().st_size > 64 * 1024**2: raise ValueError(f'Unexpected large source; review separately: {path.name}')
        yield path


def collect(phase):
    candidates = []
    for name in ('code', 'third_party', 'configs/methods', 'configs/models', 'configs/eval', 'configs/protocol', 'configs/p0', 'configs/matrix', 'configs/cache', 'configs/datasets/v3'):
        candidates.extend(source_files(ROOT / name) or [])
    for path in (ROOT / 'configs').glob('*.json'): candidates.append(path)
    for path in (ROOT / 'docs').glob('*'):
        if path.is_file() and path.suffix in ('.md', '.json') and not path.name.startswith(PRIVATE_DOC_PREFIXES): candidates.append(path)
    for pattern in AUDIT_ROOT_GLOBS: candidates.extend((ROOT / 'audit').glob(pattern))
    for name in AUDIT_DIRS: candidates.extend(source_files(ROOT / 'audit' / name) or [])
    for name in RESULT_DIRS: candidates.extend(source_files(ROOT / 'results' / name) or [])
    # Only canonical phase mirrors are selected. cache_inputs and private
    # controller/log trees are deliberately not publication sources.
    for base in (ROOT, EVIDENCE):
        candidates.extend(source_files(base / 'results' / phase) or [])
        candidates.extend(source_files(base / 'results' / (phase + '_smoke')) or [])
        scored = base / 'results/scored'
        if scored.exists():
            for folder in scored.iterdir():
                if folder.is_dir() and (folder.name == phase or folder.name.startswith(phase + '_')):
                    candidates.extend(source_files(folder) or [])
        for folder in ('results/matrix', 'results/comparisons'):
            for path in source_files(base / folder) or []:
                if phase.lower() in path.as_posix().lower(): candidates.append(path)
    selected, duplicates = {}, []
    for path in sorted(set(candidates), key=lambda p: str(p).lower()):
        origin = ROOT if inside(path, ROOT) else EVIDENCE if inside(path, EVIDENCE) else None
        if origin is None: raise ValueError('Junction/symlink points outside the two authorized source roots')
        relative = path.relative_to(origin).as_posix()
        if phase == 'p1' and relative in P1_OUT_OF_SCOPE_SOURCE: continue
        if path.name.endswith('_artifact_pointer.json') or path.name == 'artifact_pointer.json': continue
        raw = path.read_bytes()
        if relative in selected:
            prior = selected[relative]
            if prior['raw'] != raw: raise ValueError(f'Differing copies of one frozen public path: {relative}')
            duplicates.append(relative); continue
        selected[relative] = {'path': path, 'origin': 'project' if origin == ROOT else 'evidence_mirror', 'raw': raw,
                              'source_sha256': digest(raw)}
    return selected, duplicates


class Sanitizer:
    def __init__(self):
        # Exact prefixes come from ordinary local path metadata; no credential
        # helper, endpoint file, environment secrets or remote connection opens.
        replacements = [(str(ROOT), '${PROJECT_ROOT}'), (str(EVIDENCE), '${EVIDENCE_ROOT}')]
        for entry in (ROOT / 'configs/models').glob('*native*v7.json'):
            obj = load(entry)
            for key in ('model_path', 'output_dir'):
                value = obj.get(key, '')
                if '/baseline_matrix_v3/' in value:
                    replacements.append((value.split('/baseline_matrix_v3/', 1)[0] + '/baseline_matrix_v3', '${DATA_ROOT}'))
        # Both forms are needed for JSON-escaped Windows strings after parsing.
        replacements += [(str(ROOT.parent / 'TrajKV'), '${FROZEN_TRAJKV_ROOT}')]
        self.rules = []
        for before, after in sorted(set(replacements), key=lambda x: -len(x[0])):
            normal = before.replace('\\', '/')
            self.rules.append((re.compile(re.escape(normal).replace('/', r'[\\/]'), re.I), after))
        self.counts = Counter()

    def string(self, value):
        if '/' not in value and '\\' not in value:
            return value
        for rule, alias in self.rules:
            value, n = rule.subn(lambda _: alias, value); self.counts[alias] += n
        value, n = re.subn(r'/media/[^/\s"\']+/[0-9a-fA-F-]{16,}', '${DATA_DISK}', value)
        self.counts['${DATA_DISK}'] += n
        value, n = re.subn(r'(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s"\']+', '${LOCAL_HOME}', value)
        self.counts['${LOCAL_HOME}'] += n
        value, n = re.subn(r'/home/[^/\s"\']+', '${SYSTEM_HOME}', value)
        self.counts['${SYSTEM_HOME}'] += n
        return value

    def tree(self, value, field=None):
        if field in DATA_FIELDS:
            return value
        if isinstance(value, str): return self.string(value)
        if isinstance(value, list): return [self.tree(x) for x in value]
        if isinstance(value, dict): return {self.string(k): self.tree(v, k) for k, v in value.items()}
        return value

    def transform(self, relative, raw):
        suffix = Path(relative).suffix.lower()
        if suffix == '.npy': return raw, []
        text = raw.decode('utf-8-sig')
        line_map = []
        if suffix == '.json':
            # Formatting is retained whenever no path changes are required.
            obj = json.loads(text); changed = self.tree(obj)
            body = raw if changed == obj else json_bytes(changed)
        elif suffix == '.jsonl':
            pieces = []
            for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
                if not line.strip(): pieces.append(line); continue
                obj = json.loads(line); changed = self.tree(obj)
                public = line if obj == changed else (json.dumps(changed, ensure_ascii=False) + '\n').encode('utf-8')
                pieces.append(public)
                line_map.append({'line': line_number, 'id': obj.get('id'), 'source_sha256': digest(line), 'public_sha256': digest(public)})
            body = b''.join(pieces)
        elif suffix == '.csv':
            rows = list(csv.reader(io.StringIO(text)))
            if not rows:
                body = raw
            else:
                columns = rows[0]
                changed = [columns] + [[value if (i < len(columns) and columns[i] in DATA_FIELDS) else self.string(value)
                                        for i, value in enumerate(row)] for row in rows[1:]]
                if changed == rows:
                    body = raw
                else:
                    stream = io.StringIO(newline='')
                    csv.writer(stream, lineterminator='\r\n' if b'\r\n' in raw else '\n').writerows(changed)
                    body = stream.getvalue().encode('utf-8')
        else:
            changed = self.string(text)
            body = raw if changed == text else changed.encode('utf-8')
        return body, line_map


def security_findings(relative, body):
    if Path(relative).suffix == '.npy': return []
    text = body.decode('utf-8-sig')
    findings = []
    rules = {
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'github_credential': r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b',
        'hf_credential': r'\bhf_[A-Za-z0-9]{28,}\b',
        'signed_download_url': r'(?i)(?:X-Amz-(?:Signature|Credential)|[?&](?:Policy|Signature)=|X-Goog-Signature=)',
        'ssh_endpoint': r'(?i)(?:ssh://|\bssh\s+-p\s+\d+\s+\w+@|\b\w+@(?:10\.|121\.195\.))',
        'literal_credential': r'(?i)["\']?(?:password|passwd|api[_-]?key|access[_-]?token)["\']?\s*[:=]\s*["\'][^"\'\s]{6,}["\']',
        'unredacted_machine_home': r'(?i)(?:[A-Z]:[\\/]+Users[\\/]+(?!\$\{)[^\\/\s"\']+|/home/(?!\$\{)[^/\s"\']+)',
        'unredacted_disk_uuid': r'/media/[^/\s"\']+/[0-9a-fA-F-]{16,}',
    }
    for label, pattern in rules.items():
        if re.search(pattern, text): findings.append(label)
    return findings


def git_selected(relative, raw):
    if Path(relative).name in ('records.jsonl', 'scored_records.jsonl'): return False
    if relative.startswith('results/p0/') and len(raw) > 2 * 1024**2: return False
    if relative.startswith('audit/') and len(raw) > 2 * 1024**2: return False
    return len(raw) <= 16 * 1024**2


def phase_status(phase, selected, require_complete):
    matrix = ROOT / f'configs/matrix/{phase.upper()}_core_matrix_v1.json'
    if not matrix.exists():
        if require_complete: raise ValueError('No declared core matrix for phase')
        return {'status': 'NO_MATRIX_ASSERTION', 'phase': phase}
    plan = load(matrix); missing, observed = [], []
    for row in plan['rows']:
        stem = Path(row['config']).stem
        raw_prefix = f'results/{phase}/{stem}'
        raw_paths = [raw_prefix + '/' + name for name in ('manifest.json', 'records.jsonl', 'complete.json')]
        if any(path not in selected for path in raw_paths): missing.append({'cell': stem, 'missing': 'raw_three_file_set'}); continue
        complete = json.loads(selected[raw_paths[2]]['raw'])
        if complete['status'] != 'PASS_RAW_COMPLETE' or complete['expected_QA'] != row['count']:
            missing.append({'cell': stem, 'missing': 'all_expected_raw_QA_pass'}); continue
        for path, key in [(raw_paths[0], 'manifest_sha256'), (raw_paths[1], 'records_jsonl_sha256')]:
            if digest(selected[path]['raw']) != complete[key]: raise ValueError(f'Original raw integrity failure: {stem}')
        summaries = [path for path in selected if path.startswith('results/scored/') and path.endswith('/' + stem + '/summary.json')]
        if len(summaries) != 1: missing.append({'cell': stem, 'missing': 'unique_scored_summary'}); continue
        summary = json.loads(selected[summaries[0]]['raw'])
        if summary.get('status') != 'PASS_SCORED_COMPLETE' or summary['expected_run_questions'] != row['count']:
            missing.append({'cell': stem, 'missing': 'complete_scoring'}); continue
        observed.append({'cell': stem, 'questions': row['count'], 'raw_status': complete['status'], 'score_status': summary['status']})
    value = {'status': 'PASS_PHASE_COMPLETE' if not missing else 'INCOMPLETE_PHASE_PREVIEW', 'phase': phase,
             'expected_cells': len(plan['rows']), 'verified_scored_cells': len(observed),
             'expected_questions': sum(row['count'] for row in plan['rows']),
             'verified_scored_questions': sum(row['questions'] for row in observed), 'missing': missing, 'observed': observed}
    if require_complete and missing: raise ValueError('Phase not complete: ' + json.dumps(missing))
    return value


README = '''# External baseline library: {phase} increment

This increment records the declared external-baseline protocol, pinned source,
technical gates and the available {phase} evidence. It does not claim completion
of later phases or a full-split score when the manifest selects a subset.

The full exact-byte payload is in the accompanying ZIP. Git contains browsable
source/configuration/tables and small manifests; bulky per-QA evidence remains
in the ZIP. Original images, model weights, compiler caches, private controller
code and connection information are excluded. Obtain the pinned public datasets
and model revisions separately.

`PUBLIC_CONTENT_REFS.json` distinguishes original source bytes, sanitized public
archive bytes, and LF-normalized Git bytes/blob OIDs. Paths are templates such as
`${{DATA_ROOT}}`; prediction text, question text, answers, token IDs and numeric
results retain their meaning. Original SHA fields inside derived evidence still
refer to the source; resolve them through the content index rather than treating
them as a hash of the sanitized file. Per-line source/public mappings are included.

Run `python code/verify_public_export.py --root .` from an extracted namespace
to validate every public file. Add `--recompute-scores` to independently recompute
the selected offline metrics from all archived scored rows. The scorer requires
NumPy. The original strict private-source verifier is not a sanitizer-aware API.

For strict upstream byte guards use the archive: `.gitattributes` normalizes Git
text to LF, which can differ from the original checkout's CRLF bytes. Runtime
dependencies and port boundaries are stated in the model/method reports. All
matrix inference uses original FP16 weights; technical CUDA controls are separate
from benchmark quality. Prototype instrumented timing is not a controlled speed
claim. Automatic GitHub publication is a separate step after export verification.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--phase', default='p1', choices=('p1', 'p2', 'p3'))
    parser.add_argument('--require-complete-phase', action='store_true')
    parser.add_argument('--tag', default='external-baselines-p1-20261120')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.stage): raise ValueError('Use a plain new stage name')
    target = (EVIDENCE / 'public_staging' / args.stage).resolve()
    if not inside(target, EVIDENCE / 'public_staging'): raise ValueError('Staging must stay on the named E evidence volume')
    if target.exists(): raise ValueError('Frozen staging exists; choose a new stage, never overwrite it')
    target.mkdir(parents=True)
    selected, duplicates = collect(args.phase)
    phase = phase_status(args.phase, selected, args.require_complete_phase)
    sanitizer = Sanitizer(); public = target / 'tree' / NAMESPACE; public.mkdir(parents=True)
    entries, unsafe, total = {}, [], 0
    for sequence, (relative, source) in enumerate(sorted(selected.items()), 1):
        raw = source['raw']; body, line_map = sanitizer.transform(relative, raw)
        blocked = security_findings(relative, body)
        if blocked:
            unsafe.append({'path': relative, 'findings': blocked}); continue
        output = public / relative; output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(body)
        lf = body.replace(b'\r\n', b'\n') if Path(relative).suffix != '.npy' else body
        entry = {'source_origin': source['origin'], 'source_path': relative, 'source_bytes': len(raw),
                 'source_sha256': source['source_sha256'], 'public_bytes': len(body), 'public_sha256': digest(body),
                 'transformation': 'EXACT' if raw == body else 'DERIVED_PATH_TEMPLATES',
                 'git_lf': git_identity(lf), 'expected_in_git': git_selected(relative, body)}
        if line_map and relative.startswith('results/'): entry['line_identities'] = line_map
        entries[relative] = entry; total += len(body)
        if sequence % 100 == 0:
            print(json.dumps({'status': 'EXPORTING', 'processed_files': sequence, 'public_bytes': total}), flush=True)
    for relative, body in {'README_PUBLIC.md': README.format(phase=args.phase.upper()).encode(),
                           '.gitattributes': b'* text=auto eol=lf\n*.npy -text\n',
                           'code/verify_public_export.py': (ROOT / 'ops/verify_public_export.py').read_bytes()}.items():
        if relative in entries: raise ValueError('Generated path conflicts with a source file')
        output = public / relative; output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(body)
        entries[relative] = {'source_origin': 'generated_public_helper', 'source_path': None, 'source_bytes': None,
                             'source_sha256': None, 'public_bytes': len(body), 'public_sha256': digest(body),
                             'transformation': 'GENERATED_PUBLIC_HELPER', 'git_lf': git_identity(body.replace(b'\r\n', b'\n')),
                             'expected_in_git': True}
        total += len(body)
    index = {'schema': 'baseline_matrix.public_content_refs.v1', 'namespace': NAMESPACE,
             'repository': 'https://github.com/zhouz4884-coder/TrajKV', 'release_tag': args.tag,
             'created_utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
             'source_hash_fields_inside_derived_evidence': 'original immutable source identities, resolved via this map',
             'text_normalization': 'Archive preserves derived exact bytes; Git text normalizes CRLF to LF',
             'private_paths_in_index': False, 'files': entries,
             'excluded_future_unverified_source': sorted(P1_OUT_OF_SCOPE_SOURCE) if args.phase == 'p1' else [],
             'git_stage_plan': {'namespace': NAMESPACE, 'force_paths': [f'{NAMESPACE}/{p}' for p, e in entries.items() if e['expected_in_git']],
                                'plus_generated_index': f'{NAMESPACE}/PUBLIC_CONTENT_REFS.json',
                                'verify_actual_tree_and_blob_bodies': True},
             'publication_binding': 'Publisher receipt must bind fixed commit/tree/index blob OID and archive SHA; export is not upload'}
    (public / 'PUBLIC_CONTENT_REFS.json').write_bytes(json_bytes(index))
    private = {'status': 'BLOCKED_UNSAFE_SELECTED_FILES' if unsafe else 'PASS_SAFE_EXPORT',
               'stage': args.stage, 'phase': phase, 'selected_files': len(selected), 'public_files': len(entries),
               'normalization_counts': dict(sanitizer.counts), 'unsafe_omitted': unsafe, 'duplicate_exact_paths': duplicates,
               'source_files_still_match': all(v['path'].read_bytes() == v['raw'] for v in selected.values()),
               'github_mutation_performed': False}
    if not private['source_files_still_match']:
        private['status'] = 'BLOCKED_SOURCE_CHANGED_DURING_EXPORT'
    (target / 'PRIVATE_EXPORT_AUDIT.json').write_bytes(json_bytes(private))
    if unsafe or not private['source_files_still_match']:
        print(json.dumps({'status': private['status'], 'stage': args.stage, 'unsafe_paths': unsafe, 'github_mutation_performed': False}))
        raise SystemExit(2)
    archive = target / f'{args.stage}.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(public.rglob('*')):
            if path.is_file(): zipped.write(path, (Path(NAMESPACE) / path.relative_to(public)).as_posix())
    delivery = {'status': 'PREPARED_NOT_PUBLISHED', 'schema': 'baseline_matrix.public_delivery.v1',
                'stage': args.stage, 'phase': phase, 'namespace': NAMESPACE, 'release_tag': args.tag,
                'archive': archive.name, 'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive.read_bytes()),
                'content_index_sha256': digest((public / 'PUBLIC_CONTENT_REFS.json').read_bytes()),
                'content_index': 'tree/' + NAMESPACE + '/PUBLIC_CONTENT_REFS.json',
                'public_files': len(entries), 'public_payload_bytes': total,
                'expected_git_payload_files': sum(e['expected_in_git'] for e in entries.values()),
                'source_files_still_match': True, 'github_mutation_performed': False}
    (target / 'DELIVERY.json').write_bytes(json_bytes(delivery))
    print(json.dumps(delivery))


if __name__ == '__main__': main()
