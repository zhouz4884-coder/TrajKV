"""Prepare a plan-driven Round7 public increment locally, without Git/network writes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from datetime import datetime, timezone
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/a2'), str(ROOT / 'code/analysis')]
from prepare_public_snapshot import Exporter
from prepare_round5_delivery import secret_scan, zip_info
from verify_round7_evidence import quality, calibration, local_path
from round7_content_index import build as build_content_index, safe_name

SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.svg', '.png', '.pdf', '.log'}
PRIVATE = {'chat.md', 'cha.md', 'AGENTS.md', 'remote.py', 'check_github_access.py'}
PUBLICATION_TOOLS = {'prepare_round7_delivery.py', 'publish_round7_update.py', 'verify_round7_increment.py',
    'verify_round7_evidence.py', 'round7_content_index.py', 'prepare_public_snapshot.py', 'prepare_round5_delivery.py'}


class Round7Exporter(Exporter):
    """Also redact bare mount roots and known routing paths inside metadata text.

    Human answer/question content retains ordinary IP/numeric strings. Only
    exact known private project/mount roots are removed in protected fields.
    """
    def __init__(self, destination):
        super().__init__(destination)
        extra = []
        for original, _ in self.roots:
            if original.startswith('/media/') and '/VLAtest/TrajKV' in original:
                extra.append((original.split('/VLAtest/TrajKV', 1)[0], '/path/to/storage'))
        for spelling in {str(ROOT.parent), ROOT.parent.as_posix(), str(ROOT.parent).replace('\\', '\\\\')}:
            extra.append((spelling, '/path/to/research'))
        self.roots = sorted(set(self.roots + extra), key=lambda pair: -len(pair[0]))

    def clean_string(self, text, protect=False):
        text = super().clean_string(text, protect)
        if protect:
            for original, target in self.roots:
                text = text.replace(original, target)
        return text


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def allowed(path):
    path = Path(path)
    parts = path.relative_to(ROOT).parts
    if set(parts) & {'.git', '__pycache__', 'images', 'models', 'weights', '.ssh', 'private', 'chats', 'conversations'}:
        return False
    name = path.name
    if name in PRIVATE or re.search(r'^chat|^DS(?:[_. -]|\d|$)|credential|auth[_-]?helper|connection', name, re.I):
        return False
    if name.startswith('EXECUTION_CHECKPOINT') or path.suffix.lower() not in SUFFIXES:
        return False
    if tuple(parts[:2]) == ('code', 'publication') and name not in PUBLICATION_TOOLS:
        return False
    if path.suffix.lower() in {'.png', '.pdf'} and parts[0] != 'figures':
        return False
    return True


def find_fingerprint(folder, expected):
    matches = [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl') and sha(p) == expected]
    require(matches, 'Missing original frozen fingerprint: ' + expected)
    return matches[0]


def preflight(plan):
    receipt_path = local_path(plan['execution_receipt'])
    receipt = load(receipt_path)
    research_ready = (receipt.get('status') == 'RESEARCH_COMPLETE_READY_FOR_PUBLICATION'
        and receipt.get('research_execution_completed') is True
        and receipt.get('execution_goal_completed') is False and receipt.get('github_sync_pending') is True)
    legacy_complete = (receipt.get('status') == 'COMPLETE'
        and (receipt.get('execution_goal_completed') is True or receipt.get('stage_complete') is True))
    require(research_ready or legacy_complete,
            'Receipt must establish research completion and explicitly distinguish pending GitHub delivery/global goal')
    report = local_path(plan['final_report'])
    intro = local_path(plan['readme_entry_file'])
    require(report.is_file() and report.stat().st_size > 0 and intro.is_file() and intro.stat().st_size > 0,
            'Final report and concrete README entry must exist before preparation')
    verification_path = local_path(plan['evidence_verification'])
    verification = load(verification_path)
    require(verification.get('status') == 'PASS', 'Independent local evidence verification must pass')
    quality_checks = []
    required = {receipt_path, report, intro, verification_path}
    specs = []
    analysis_entries = list(verification['quality_analyses'])
    known = {local_path(entry['summary']).resolve() for entry in analysis_entries}
    if plan.get('include_completed_development_analyses', True):
        for path in sorted((ROOT / 'results/round7_v1').rglob('summary.json')):
            if path.resolve() in known or not (path.parent / 'scores.csv').is_file():
                continue
            value = load(path)
            if (value.get('status') == 'COMPLETE' and value.get('audit', {}).get('status') == 'PASS'
                    and value.get('audit', {}).get('runs') and value.get('samples_sha256') and value.get('references_sha256')):
                analysis_entries.append(dict(summary=relative(path), summary_sha256=sha(path)))
                known.add(path.resolve())
    for entry in analysis_entries:
        summary_path = local_path(entry['summary'])
        require(sha(summary_path) == entry['summary_sha256'], 'Quality summary changed after independent verification')
        check = quality(summary_path)
        quality_checks.append(check)
        summary = load(summary_path)
        sample = find_fingerprint('configs/a2', summary['samples_sha256'])
        refs = find_fingerprint('configs/data', summary['references_sha256'])
        required.update({sample, refs, summary_path, summary_path.parent / 'scores.csv'})
        runs = []
        for run in summary['audit']['runs']:
            folder = local_path(run['run'])
            manifest = load(folder / 'manifest.json')
            for name in ('manifest.json', 'complete.json', 'records.jsonl'):
                required.add(folder / name)
            for name in manifest.get('external_codec_inputs', {}):
                required.add(local_path(name))
            source = local_path(manifest['config']['arguments']['samples'])
            required.add(source)
            runs.append(relative(folder / 'records.jsonl'))
        specs.append(dict(dataset=check['dataset'], analysis_role=summary.get('analysis_role'),
            summary=relative(summary_path), scores=relative(summary_path.parent / 'scores.csv'),
            samples=relative(sample), references=relative(refs), raw_records=runs,
            selected_variants=[row['variant'] for row in summary['table']],
            analysis_method_filter=summary.get('analysis_method_filter', {})))
    require(set(plan.get('required_quality_domains', ['doc', 'text'])) <= {x['dataset'] for x in specs},
            'Required quality domains are missing')
    calibration_checks = []
    for entry in verification.get('calibration_diagnostics', []):
        check = calibration(local_path(entry['path']))
        calibration_checks.append(check)
    for entry in verification.get('failed_smokes', []):
        for item in entry['files']:
            require(sha(local_path(entry['path']) / item['path']) == item['sha256'], 'Preserved failed-smoke evidence changed')
    for item in plan.get('additional_checks', []):
        path = local_path(item['path'])
        require(load(path).get('status') in item.get('accepted_statuses', ['PASS', 'COMPLETE']), 'Incomplete additional stage check')
        required.add(path)
    if plan.get('method_freeze'):
        frozen_path = local_path(plan['method_freeze'])
        frozen = load(frozen_path)
        for item in frozen.get('frozen_files', []):
            require(sha(local_path(item['path'])) == item['sha256'], 'Frozen method/source input changed')
            if local_path(item['path']).is_relative_to(ROOT):
                required.add(local_path(item['path']))
        required.add(frozen_path)
    checks = dict(status='PASS', execution_receipt=relative(receipt_path), execution_receipt_sha256=sha(receipt_path),
        evidence_verification=relative(verification_path), evidence_verification_sha256=sha(verification_path),
        quality_analyses=specs, numeric_recounts=quality_checks, calibration_diagnostics=calibration_checks,
        failed_smokes=verification.get('failed_smokes', []),
        research_execution_completed=True, github_sync_pending_at_preparation=research_ready,
        execution_goal_completed_at_preparation=bool(receipt.get('execution_goal_completed')),
        analysis_views_may_share_raw_runs=True,
        scope='completed free-running quality, cached-prefix diagnostics and failed/partial smokes remain separate; overlapping analysis views do not create independent data; research completion precedes publication/global goal completion')
    return checks, required


def transformed(exporter, path, name, failed=False):
    if path.suffix.lower() != '.jsonl':
        raw = path.read_bytes()
        if path.suffix.lower() in {'.txt', '.log'} and raw.startswith((b'\xff\xfe', b'\xfe\xff')):
            data = exporter.clean_string(raw.decode('utf-16')).encode('utf-8')
        else:
            data = exporter.transform(path, Path(name))
        secret_scan(data, name)
        return data, {}
    output, complete_rows, partial_rows = [], 0, 0
    text = path.read_bytes().decode('utf-8-sig', errors='replace' if failed else 'strict')
    for line in text.splitlines(keepends=True):
        if not line.strip():
            output.append(line)
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            require(failed, 'Incomplete JSONL outside an explicitly preserved failed run: ' + name)
            output.append(exporter.clean_string(line)); partial_rows += 1
        else:
            clean = exporter.clean_object(value)
            output.append(line if clean == value else json.dumps(clean, ensure_ascii=False) + '\n')
            complete_rows += 1
    data = ''.join(output).encode('utf-8')
    secret_scan(data, name)
    return data, dict(complete_json_rows=complete_rows, unparsed_partial_rows=partial_rows,
        parse_scope='failed/partial evidence, excluded from quality' if failed else 'complete JSONL')


def prepare(args):
    plan = load(args.plan)
    require(plan.get('schema') == 'trajkv.round7.publication-plan.v1', 'Explicit Round7 publication plan required')
    require(not args.output.exists(), 'Use a new immutable preparation directory')
    require(args.output.resolve().is_relative_to(ROOT / 'artifacts'), 'Prepare on the project data disk under artifacts/')
    checks, required = preflight(plan)
    required.add(args.plan.resolve())
    baseline = load(args.public_baseline / 'PUBLIC_EXPORT_MANIFEST.json')
    candidates = set(required)
    for folder in ('code', 'configs', 'docs', 'figures', 'audit/a2'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and allowed(path) and baseline.get('files', {}).get(relative(path), {}).get('source_sha256') != sha(path):
                candidates.add(path)
    for folder in [ROOT / 'results/round7_v1', *[local_path(x) for x in plan.get('extra_roots', [])]]:
        require(folder.is_relative_to(ROOT), 'Extra evidence root must stay in this project')
        for path in folder.rglob('*'):
            if path.is_file() and allowed(path):
                candidates.add(path)
    for path in required:
        require(path.is_file() and allowed(path), 'Required public evidence is missing/prohibited: ' + str(path))
    args.output.mkdir(parents=True)
    payload = args.output / 'payload'; payload.mkdir()
    exporter = Round7Exporter(payload)
    failed_roots = [local_path(x['path']).resolve() for x in checks['failed_smokes']]
    failed_roots += [local_path(x).resolve() for x in plan.get('failed_evidence_roots', [])]
    archive_path = args.output / plan.get('archive_name', 'round7_increment_20261116.zip')
    entries, runtime = {}, []
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(candidates):
            name = relative(path)
            original_sha = sha(path)
            failed = any(path.resolve().is_relative_to(folder) for folder in failed_roots)
            # Completed primary dependencies must always parse completely. Old
            # aborted diagnostics may have only a partial final line; archive
            # them with explicit scope instead of inventing a complete record.
            formal_raw = {value for spec in checks['quality_analyses'] for value in spec['raw_records']}
            if path.suffix == '.jsonl' and name.startswith('results/') and name not in formal_raw:
                folder = path.parent
                if not (folder / 'complete.json').is_file():
                    failed = True
            data, extra = transformed(exporter, path, name, failed)
            require(sha(path) == original_sha, 'Source changed during preparation; preserve this attempt and use a new version: ' + name)
            browsable = not (name.startswith('results/') and path.suffix.lower() in {'.jsonl', '.log'}) and len(data) < 90 * 1024 * 1024
            if browsable:
                target = payload / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            archive.writestr(zip_info(name), data)
            entries[name] = dict(source_path=name, source_sha256=original_sha, source_bytes=path.stat().st_size,
                public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=browsable,
                role='preserved failed/partial smoke; not primary performance' if failed else
                     'cached-prefix calibration diagnostic' if any(path.resolve().is_relative_to(local_path(x['path'])) for x in checks['calibration_diagnostics']) else
                     'derived research source/configuration/result evidence', **extra)
        seen_runtime = set()
        for collection_id, source_root in enumerate(args.runtime_sources):
            require(source_root.is_dir(), 'Supplied runtime source collection is missing')
            collection = source_root / 'COLLECTION.json'
            if collection.is_file():
                require(load(collection).get('status') in ('PASS', 'COMPLETE'), 'Runtime source collection did not pass')
            for metadata_name in ('runtime_index.json', 'COLLECTION.json'):
                metadata_path = source_root / metadata_name
                if not metadata_path.is_file():
                    continue
                name = f'runtime_sources/collection{collection_id}_' + metadata_name
                data = exporter.transform(metadata_path, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=sha(metadata_path), source_bytes=metadata_path.stat().st_size,
                    public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=False,
                    role='per-job source inventory and narrow source mapping; inventory presence is not proof of execution')
            for path in sorted(source_root.rglob('*.py')):
                if path.name in PRIVATE or re.search(r'credential|auth[_-]?helper|connection', path.name, re.I):
                    continue
                text = path.read_text(encoding='utf-8-sig', errors='replace')
                if any(marker in text for marker in ('contractkv_ssh_credential', 'win32cred', 'CredReadW', 'CredRead(')):
                    continue
                original = sha(path)
                if re.fullmatch(r'[0-9a-f]{64}', path.stem):
                    require(path.stem == original, 'Runtime blob filename differs from original source SHA')
                if original in seen_runtime:
                    continue
                seen_runtime.add(original)
                name = 'runtime_sources/' + original + '.py'
                data = exporter.transform(path, Path(name)); secret_scan(data, name)
                archive.writestr(zip_info(name), data)
                entries[name] = dict(source_sha256=original, source_bytes=path.stat().st_size,
                    public_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), browsable=False,
                    role='original runtime source version deduplicated by private source SHA; public bytes are path-redacted')
                runtime.append(dict(source_sha256=original, content_path=name))
        index = dict(schema='trajkv.round7.public.increment.v1', status='PASS',
            generated_at_utc=datetime.now(timezone.utc).isoformat(), files=entries, checks=checks,
            publication_plan=plan, publication_plan_sha256=sha(args.plan),
            baseline_manifest_sha256=sha(args.public_baseline / 'PUBLIC_EXPORT_MANIFEST.json'),
            runtime_source_blobs=runtime,
            exclusions='private chats/checkpoints/authentication/connection helpers, original images, model weights, all tensor/checkpoint/moments.pt files',
            hash_scope='source SHA fixes preserved original bytes; public SHA fixes derived archive bytes; fixed content_ref and LF Git identities are recorded separately',
            snapshot_scope='a recorded source inventory does not prove every module executed; historical source versions are retained when supplied',
            public_write_performed=False)
        encoded = (json.dumps(exporter.clean_object(index), indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        secret_scan(encoded, 'ROUND7_SOURCE_INDEX.json')
        archive.writestr(zip_info('ROUND7_SOURCE_INDEX.json'), encoded)
    index_path = args.output / 'ROUND7_SOURCE_INDEX.json'; index_path.write_bytes(encoded)
    content_path = args.output / 'ROUND7_CONTENT_REFS.json'
    content = build_content_index(SimpleNamespace(output=content_path, source_index=index_path, payload=payload, archive=archive_path))
    delivery = dict(status='PREPARED_LOCAL_ONLY', archive=archive_path.name, archive_sha256=sha(archive_path),
        archive_bytes=archive_path.stat().st_size, source_index_sha256=sha(index_path), content_index_sha256=sha(content_path),
        files=len(entries), browsable_files=sum(x['browsable'] for x in entries.values()),
        force_add_metadata_paths=content['git_stage_plan']['force_paths'], public_write_performed=False,
        source_plan=str(args.plan.resolve()), source_plan_sha256=sha(args.plan))
    (args.output / 'DELIVERY.json').write_text(json.dumps(delivery, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in delivery.items() if key != 'force_add_metadata_paths'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--public-baseline', type=Path, default=ROOT.parent / 'TrajKV_public_20261114')
    p.add_argument('--runtime-sources', nargs='*', type=Path, default=[])
    args = p.parse_args()
    args.plan = args.plan.resolve(); args.public_baseline = args.public_baseline.resolve()
    if args.preflight_only:
        checks, required = preflight(load(args.plan))
        print(json.dumps(dict(status='PASS', quality_analyses=len(checks['quality_analyses']),
            research_execution_completed=checks['research_execution_completed'],
            execution_goal_completed_at_preparation=checks['execution_goal_completed_at_preparation'],
            github_sync_pending_at_preparation=checks['github_sync_pending_at_preparation'],
            required_input_files=len(required), writes=0, network_operations=0)))
    else:
        require(args.output, '--output required for actual local preparation')
        args.output = args.output.resolve(); prepare(args)
