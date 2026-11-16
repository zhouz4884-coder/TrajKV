"""Offline derived Round5 increment; never mutate the public checkout/publish.

Prepare only after fresh Doc/Text confirmation and fixed-output timing finish.
--verify is read-only. Old releases, receipts, images, weights, authentication
helpers and private chat are never copied or overwritten by this script.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
from prepare_public_snapshot import Exporter, SECRET

ROOT = Path(__file__).resolve().parents[2]
SAFE_TEXT = {'.py', '.json', '.jsonl', '.csv', '.md', '.svg', '.txt'}
SAFE_FIGURE = {'.png', '.pdf'}
OWNER_KEYS = ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position',
              'spectrum_experiment_reference', 'rotation_packed_reference',
              'hadamard_constant', 'axis_packed_reference', 'calib_packed_reference',
              'hybrid_packed_reference', 'kvarn_packed_reference', 'anchor_visual_auxiliary')
PRIVATE_NAMES = {'chat.md', 'cha.md', 'AGENTS.md', 'check_github_access.py', 'remote.py'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def path_under_root(path):
    value = Path(path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('Evidence must be in the source workspace: '+str(value))
    return value


def relative(path):
    return path_under_root(path).relative_to(ROOT).as_posix()


def secret_scan(data, name):
    if Path(name).suffix.lower() in SAFE_FIGURE:
        return
    text = data.decode('utf-8-sig')
    if re.search(r'-----BEGIN (?:[A-Z ]*PRIVATE KEY)-----', text):
        raise ValueError('Private key marker in derived export: '+name)
    for match in SECRET.finditer(text):
        value = match.group(1) if match.lastindex else ''
        if value.lower() not in ('none', 'null', 'redacted', 'your_token', 'your_password', 'placeholder'):
            raise ValueError('Credential pattern in derived export: '+name)
    if Path(name).suffix == '.py':
        ast.parse(text, filename=name)


def allowed(path):
    parts = set(path.relative_to(ROOT).parts)
    if parts & {'.git', '__pycache__', 'job_sources', 'analysis_sources', 'images', 'models', 'weights', 'publication'}:
        return False
    if path.name in PRIVATE_NAMES or path.name.startswith('EXECUTION_CHECKPOINT'):
        return False
    if re.search(r'(credential|auth[_-]?helper)', path.name, re.I):
        return False
    if path.suffix.lower() in SAFE_FIGURE and not path.is_relative_to(ROOT/'figures'):
        return False
    return path.suffix.lower() in SAFE_TEXT | SAFE_FIGURE


def verify_sha_list(folder):
    mapping = load(folder/'SHA256.json')
    if not mapping:
        raise ValueError('Empty analysis SHA list')
    for name, expected in mapping.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(folder/name) != expected:
            raise ValueError('Analysis SHA mismatch: '+str(folder/name))
    return dict(directory=relative(folder), sha256_manifest=sha(folder/'SHA256.json'), files=len(mapping))


def quality_means(summary, scores):
    groups = defaultdict(list)
    for row in scores:
        value = float(row['score'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid exported score')
        groups[row['variant']].append(value)
    if set(groups) != {row['variant'] for row in summary['table']}:
        raise ValueError('Score/table variant set differs')
    for row in summary['table']:
        actual = sum(groups[row['variant']])/len(groups[row['variant']])
        if abs(actual-row.get('anls', row.get('accuracy'))) > 1e-10:
            raise ValueError('CSV mean differs from result table: '+row['variant'])
    return dict(methods=len(groups), rows=sum(map(len, groups.values())))


def records_bits(records):
    stores, denominators = defaultdict(float), defaultdict(float)
    for row in records:
        owned = row['owned_initial']
        stores[row['variant']] += sum(owned.get(key, 0) for key in OWNER_KEYS)+row.get('encoder_calibration_parameters_bytes', 0)
        denominators[row['variant']] += owned['visual_fp16_denominator']
    return {name: 16*amount/denominators[name] for name, amount in stores.items()}


def json_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def preflight(a):
    doc = load(a.doc_confirmation/'summary.json')
    text = load(a.text_analysis/'summary.json')
    basic = load(a.doc_analysis/'summary.json')
    marker = load(ROOT/'audit/a2/round5_confirmation_v1.json')
    for folder in (a.doc_confirmation, a.text_analysis):
        if load(folder/'complete.json')['status'] != 'COMPLETE':
            raise ValueError('Required quality analysis incomplete')
    if (doc['status'], text['status'], basic['status']) != ('COMPLETE',)*3:
        raise ValueError('Quality summaries incomplete')
    if doc['primary_candidate'] != marker['primary'] or text['primary_candidate'] != marker['primary']:
        raise ValueError('Frozen primary identity changed')
    if doc['audit']['status'] != 'PASS' or text['audit']['status'] != 'PASS':
        raise ValueError('Confirmation source/owner audit failed')
    if doc['input_summary_sha256'] != sha(a.doc_analysis/'summary.json') or doc['scores_sha256'] != sha(a.doc_analysis/'scores.csv'):
        raise ValueError('Doc confirmation does not refer to supplied scored analysis')
    if (doc['images'], doc['questions_per_variant'], doc['variants']) != (64, 256, 12) or (text['images'], text['questions_per_variant'], text['variants']) != (64, 64, 7):
        raise ValueError('Required cohort coverage differs')
    quality_specs = []
    stages = []
    for kind, summary, scores in [('doc', basic, a.doc_analysis/'scores.csv'),
                                  ('text', text, a.text_analysis/'scores.csv')]:
        check = quality_means(summary, list(csv.DictReader(scores.open(encoding='utf-8-sig'))))
        raw_paths = []
        rows = []
        for shard in (0, 1):
            parent = ROOT/f'results/round5_v1/confirm_{kind}_s{shard}'
            if load(parent/'complete.json')['status'] != 'COMPLETE':
                raise ValueError('Required new inference stage incomplete')
            folder = parent/'transfer32' if kind == 'text' else parent
            if load(folder/'complete.json')['status'] != 'COMPLETE':
                raise ValueError('Required inference child incomplete')
            raw = folder/'records.jsonl'
            raw_paths.append(relative(raw)); rows.extend(json_lines(raw))
            stages.append(dict(path=relative(folder/'complete.json'), sha256=sha(folder/'complete.json')))
        bits = records_bits(rows)
        for row in summary['table']:
            if abs(bits[row['variant']]-row['budget_bits_including_encoder_constants']) > 1e-10:
                raise ValueError('Actual record owner differs from quality table')
        quality_specs.append(dict(kind=kind, summary=relative(a.doc_analysis/'summary.json') if kind == 'doc' else relative(a.text_analysis/'summary.json'),
            scores=relative(scores), raw_records=raw_paths, checks=check))
    timing = load(a.timing/'complete.json')
    timing_summary = load(a.timing_analysis/'summary.json')
    if timing['status'] != 'COMPLETE' or timing.get('quality') != 'not_scored' or timing_summary['status'] != 'PASS':
        raise ValueError('Required fixed-output timing is incomplete')
    for key in ('output_ids_identical_each_question', 'logical_positions_identical_at_same_cap', 'quiet_verified_all_trials'):
        if timing_summary.get(key) is not True:
            raise ValueError('Timing protocol check failed')
    measurements = list(csv.DictReader((a.timing_analysis/'measurements.csv').open(encoding='utf-8-sig')))
    if len(measurements) != timing['measurement_trials'] or len(measurements) != timing_summary['measured_trials']:
        raise ValueError('Timing trial count differs')
    for row in timing_summary['table']:
        values = [float(item['session_ms']) for item in measurements if item['variant'] == row['variant']]
        if not values or abs(statistics.median(values)-row['session_ms_median']) > 1e-9:
            raise ValueError('Timing CSV median differs from table')
    stages.append(dict(path=relative(a.timing/'complete.json'), sha256=sha(a.timing/'complete.json')))
    return dict(status='PASS', phase_scope=['new Doc confirmation', 'new Text transfer', 'quiet fixed-output timing'],
        required_complete=stages, quality_analyses=quality_specs,
        timing_analysis=relative(a.timing_analysis/'summary.json'),
        analysis_sha_checks=[verify_sha_list(a.doc_confirmation), verify_sha_list(a.text_analysis)],
        timing_analysis_sha256={name: sha(a.timing_analysis/name) for name in ('summary.json', 'measurements.csv', 'table.csv')},
        frozen_marker_sha256=sha(ROOT/'audit/a2/round5_confirmation_v1.json'))


def snapshot_needs():
    """Broad recorded source lists are not execution traces; filter analysis."""
    needed = {}
    for manifest in (ROOT/'results/round5_v1').rglob('manifest.json'):
        data = load(manifest)
        sources = data.get('sources', data.get('source_sha256', {}))
        if not isinstance(sources, dict):
            continue
        for name, expected in sources.items():
            if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or not name.endswith('.py'):
                continue
            leaf = Path(name).name
            if leaf in PRIVATE_NAMES or leaf.startswith(('analyze', 'summarize', 'plot', 'prepare', 'configure', 'freeze', 'audit', 'stage', 'report')):
                continue
            rel = name.replace('\\', '/') if '/' in name.replace('\\', '/') else 'code/a2/'+name
            source = ROOT/rel
            current = sha(source) if source.is_file() else None
            if current != expected:
                item = needed.setdefault((rel, expected), dict(source=rel, expected_sha256=expected,
                    current_sha256=current, declaring_manifests=[]))
                item['declaring_manifests'].append(relative(manifest))
    return list(needed.values())


def candidates(baseline):
    old = baseline.get('files', {})
    for folder in ('code', 'configs', 'docs', 'figures', 'audit'):
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or not allowed(path):
                continue
            if folder == 'audit' and not path.is_relative_to(ROOT/'audit/a2'):
                continue
            rel = relative(path)
            if old.get(rel, {}).get('source_sha256') != sha(path):
                yield path
    for path in sorted((ROOT/'results/round5_v1').rglob('*')):
        if path.is_file() and allowed(path):
            yield path


def zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 1
    return info


def prepare(a):
    checks = preflight(a)
    baseline_path = a.public_baseline/'PUBLIC_EXPORT_MANIFEST.json'
    baseline = load(baseline_path)
    destination = a.output.resolve()
    if not destination.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Use a fresh directory on the workspace data disk under artifacts')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite a previous Round5 delivery directory')
    destination.mkdir(parents=True, exist_ok=True)
    payload = destination/'payload'
    exporter = Exporter(payload)
    paths = sorted(set(candidates(baseline)))
    raw = []
    for path in paths:
        if path.suffix == '.jsonl' and path.is_relative_to(ROOT/'results/round5_v1'):
            raw.append(path)
        else:
            exporter.add(path)
    if exporter.findings:
        raise ValueError('Derived files require credential-pattern review')
    archive_path = destination/'round5_increment_20261114.zip'
    entries = dict(exporter.entries)
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for name in sorted(entries):
            data = (payload/name).read_bytes(); secret_scan(data, name)
            archive.writestr(zip_info(name), data)
        for path in raw:
            name = relative(path)
            public_hash, count, byte_count, invalid = hashlib.sha256(), 0, 0, 0
            with archive.open(zip_info(name), 'w', force_zip64=True) as output, path.open('rb') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        original = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        invalid += 1; continue
                    if invalid:
                        raise ValueError('Malformed middle JSONL row, not an interrupted trailing tail: '+name)
                    clean = exporter.clean_object(original)
                    data = line if clean == original else (json.dumps(clean, ensure_ascii=False)+'\n').encode('utf-8')
                    secret_scan(data, name)
                    output.write(data); public_hash.update(data); count += 1; byte_count += len(data)
            entries[name] = dict(source_sha256=sha(path), public_sha256=public_hash.hexdigest(),
                source_bytes=path.stat().st_size, bytes=byte_count, rows=count,
                invalid_trailing_rows_omitted=invalid,
                role='raw execution/diagnostic rows; row counts are not independent sample counts')
        needs = snapshot_needs()
        if a.snapshot_sources:
            required = {item['expected_sha256'] for item in needs}
            for source in sorted(a.snapshot_sources.rglob('*.py')):
                source_sha = sha(source)
                if source_sha not in required:
                    continue
                name = 'runtime_sources/'+source_sha+'.py'
                if name in entries:
                    continue
                data = exporter.transform(source, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=source_sha, public_sha256=digest(data), bytes=len(data),
                    role='SHA-deduplicated original runtime source, derived path redaction')
            present = {entry['source_sha256'] for name, entry in entries.items() if name.startswith('runtime_sources/')}
            for item in needs:
                item['archived_blob'] = 'runtime_sources/'+item['expected_sha256']+'.py' if item['expected_sha256'] in present else None
        index = dict(schema='trajkv.round5.public.increment.v1', status='PASS',
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            role='offline derived Round5 increment; no GitHub or existing public-checkout write',
            baseline_manifest_sha256=sha(baseline_path), files=entries, checks=checks,
            source_snapshot_needed=needs,
            snapshot_scope='recorded module-list mismatches, not proof every listed module executed; final current-source identities are separately frozen/audited',
            exclusions='original images, weights, chat, credentials/authentication helpers, all tensor/moment artifacts, old release ZIPs/receipts and duplicate old raw evidence',
            hash_scope='source SHA identifies the preserved private original; public SHA identifies this derived content; benchmark answers and numerical metrics remain unchanged')
        data = (json.dumps(exporter.clean_object(index), indent=2, ensure_ascii=False)+'\n').encode('utf-8')
        secret_scan(data, 'ROUND5_SOURCE_INDEX.json')
        archive.writestr(zip_info('ROUND5_SOURCE_INDEX.json'), data)
    (destination/'ROUND5_SOURCE_INDEX.json').write_bytes(data)
    receipt = dict(status='PREPARED', archive=archive_path.name, archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha(archive_path), index_sha256=digest(data), files=len(entries),
        public_write_performed=False, prior_release_modified=False,
        source_snapshot_requests=len(needs))
    (destination/'DELIVERY.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    verified = verify(destination)
    (destination/'VERIFICATION.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
    print(json.dumps(dict(receipt, verification=verified['status'])))


def verify(destination):
    """Read-only ZIP/privacy/syntax/numerical verification, no source mutation."""
    receipt = load(destination/'DELIVERY.json')
    archive_path = destination/receipt['archive']
    if sha(archive_path) != receipt['archive_sha256'] or sha(destination/'ROUND5_SOURCE_INDEX.json') != receipt['index_sha256']:
        raise ValueError('Delivery archive/index SHA mismatch')
    index = load(destination/'ROUND5_SOURCE_INDEX.json')
    syntax, scanned = 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        if archive.read('ROUND5_SOURCE_INDEX.json') != (destination/'ROUND5_SOURCE_INDEX.json').read_bytes():
            raise ValueError('Inner/outer source index differs')
        if set(archive.namelist()) != set(index['files']) | {'ROUND5_SOURCE_INDEX.json'}:
            raise ValueError('Archive member inventory differs')
        for name, entry in index['files'].items():
            parts = set(name.split('/'))
            suffix = Path(name).suffix.lower()
            if (name.startswith('/') or '\\' in name or '..' in parts or ':' in name.split('/')[0]
                    or Path(name).name in PRIVATE_NAMES or parts & {'images', 'models', 'weights', '.git', '__pycache__'}
                    or suffix not in SAFE_TEXT | SAFE_FIGURE
                    or (suffix in SAFE_FIGURE and not name.startswith('figures/'))
                    or re.search(r'(credential|auth[_-]?helper)', Path(name).name, re.I)):
                raise ValueError('Prohibited archive member')
            data = archive.read(name)
            if digest(data) != entry['public_sha256'] or len(data) != entry['bytes']:
                raise ValueError('Public content SHA/size mismatch: '+name)
            secret_scan(data, name); scanned += 1; syntax += name.endswith('.py')
            if not name.startswith('runtime_sources/'):
                original = ROOT/name
                if not original.is_file() or sha(original) != entry['source_sha256']:
                    raise ValueError('Preserved source changed after increment preparation: '+name)
        numeric = []
        for spec in index['checks']['quality_analyses']:
            summary = json.loads(archive.read(spec['summary']))
            scores = list(csv.DictReader(io.StringIO(archive.read(spec['scores']).decode('utf-8-sig'))))
            numeric.append(dict(kind=spec['kind'], **quality_means(summary, scores)))
            records = [json.loads(line) for name in spec['raw_records'] for line in archive.read(name).splitlines() if line.strip()]
            bits = records_bits(records)
            for row in summary['table']:
                if abs(bits[row['variant']]-row['budget_bits_including_encoder_constants']) > 1e-10:
                    raise ValueError('Derived public owner bit table mismatch')
    return dict(status='PASS', files=scanned, python_ast_files=syntax,
        credential_findings=0, numerical_recomputations=numeric,
        archive_sha256=receipt['archive_sha256'], source_index_sha256=receipt['index_sha256'],
        scope='read-only verification of derived archive; no network, Git or filesystem writes')


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/round5_publication_assets')
    p.add_argument('--public-baseline', type=Path, default=ROOT.parent/'TrajKV_public_20261114')
    p.add_argument('--doc-analysis', type=Path, default=ROOT/'results/round5_v1/doc_confirm64_analysis')
    p.add_argument('--doc-confirmation', type=Path, default=ROOT/'results/round5_v1/doc_confirmation_summary')
    p.add_argument('--text-analysis', type=Path, default=ROOT/'results/round5_v1/text_confirm64_analysis')
    p.add_argument('--timing', type=Path, default=ROOT/'results/round5_v1/quiet_timing')
    p.add_argument('--timing-analysis', type=Path, default=ROOT/'results/round5_v1/quiet_timing_analysis')
    p.add_argument('--snapshot-sources', type=Path)
    a = p.parse_args()
    if a.self_test:
        result = quality_means(dict(table=[dict(variant='synthetic', accuracy=.75)]),
            [dict(variant='synthetic', score='1'), dict(variant='synthetic', score='.5')])
        bits = records_bits([dict(variant='synthetic', owned_initial=dict(
            visual_base=40, visual_residual=10, selector_metadata=2,
            visual_position=4, visual_fp16_denominator=128), encoder_calibration_parameters_bytes=8)])
        if bits['synthetic'] != 8. or result != dict(methods=1, rows=2):
            raise RuntimeError('Synthetic mean/owner accounting failed')
        rejected = False
        try:
            secret_scan(b'sk-'+b'a'*30, 'synthetic.txt')
        except ValueError:
            rejected = True
        if not rejected:
            raise RuntimeError('Synthetic credential pattern was not rejected')
        secret_scan(b'x = 1\n', 'synthetic.py')
        print(json.dumps(dict(status='PASS', writes=0, numeric_mean=True,
            complete_owner=True, secret_pattern_rejected=True, python_ast=True)))
        return
    if a.verify:
        print(json.dumps(verify(a.output.resolve()))); return
    for key in ('doc_analysis', 'doc_confirmation', 'text_analysis', 'timing', 'timing_analysis'):
        setattr(a, key, path_under_root(getattr(a, key)))
    prepare(a)


if __name__ == '__main__':
    main()
