"""Create or verify an immutable methods/artifacts/pools confirmation freeze.

Creation must precede R7 confirmation-reference materialization. A freeze pins
identities and declared choices; it does not itself assert experimental merit.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_POOL = ROOT / 'audit/a2/round7_doc_confirm128_candidate_pool_v1.json'
TEXT_POOL = ROOT / 'audit/a2/round7_text_confirm128_membership_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    normalized = str(value).replace('\\', '/')
    if '/TrajKV/' in normalized:
        return ROOT / normalized.split('/TrajKV/', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_entry(path, purpose):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path)
    return dict(path=name, sha256=sha(path), bytes=path.stat().st_size, purpose=purpose)


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(path):
    freeze = read(path)
    require(freeze.get('schema') == 'trajkv.round7.method-freeze.v1', 'Wrong freeze schema')
    for item in freeze['frozen_files']:
        require(sha(local_path(item['path'])) == item['sha256'], 'Frozen input changed: ' + item['path'])
    require(freeze.get('confirmation_references_materialized_at_creation') is False,
            'Freeze did not precede confirmation-reference materialization')
    require(freeze.get('selected_variants'), 'Empty method freeze')
    print(json.dumps(dict(status='PASS', freeze=str(path), freeze_sha256=sha(path),
        frozen_files=len(freeze['frozen_files']), variants=list(freeze['selected_variants']), GPU_used=False,
        scope='unchanged pinned input identities; no inference or statistical claim')))


def create(args):
    require(args.configs, 'At least one concrete config is required for creation')
    for relative in ('configs/data/round7_doc_confirm128_references.jsonl', 'configs/data/round7_text_confirm128_references.json'):
        require(not (ROOT / relative).exists(), 'Confirmation refs already materialized; cannot create a pre-answer freeze: ' + relative)
    require(not args.output.exists(), 'Refusing to replace an existing freeze')
    doc, text = read(DOC_POOL), read(TEXT_POOL)
    require(doc.get('answers_read') is False and doc.get('question_text_read') is False
            and len(doc['candidates128']) == 128, 'Doc candidate identities were not frozen without question/answer access')
    require(text.get('answers_read') is False and len(text['images_manifest']) == 128, 'Text identities were not frozen without labels')
    chosen, config_rows, files = {}, [], {}
    for config_path in args.configs:
        config = read(config_path)
        config_rows.append(config)
        item = file_entry(config_path, 'concrete generation/method configuration')
        files[item['path']] = item
        for variant in config['variants']:
            name = variant['name']
            if args.selected_variants and name not in args.selected_variants:
                continue
            require(name not in chosen or chosen[name] == variant, 'Same selected name has conflicting configs')
            chosen[name] = variant
    require(chosen and (not args.selected_variants or set(chosen) == set(args.selected_variants)), 'Selected variants missing from concrete configs')
    for variant in chosen.values():
        for key in ('direction_book_path', 'metadata_factor_path', 'key_metric_path', 'functional_v_factor_path',
                    'interaction_policy_path', 'allocation_path', 'policy_path'):
            if variant.get(key):
                item = file_entry(local_path(variant[key]), 'deployed/calibrated method artifact: ' + key)
                files[item['path']] = item
    for path in args.artifacts:
        item = file_entry(path, 'calibration/policy artifact; cached-prefix diagnostics are not E2E quality or timing')
        files[item['path']] = item
    methods = {variant['method'] for variant in chosen.values()}
    common = ['code/a2/masked_bank.py', 'code/a2/metadata_direction_bank.py', 'code/a2/learned_direction_bank.py',
              'code/a2/affine_rate_bank.py', 'code/a2/quantizers.py', 'code/a2/fast_wht.py',
              'code/a2/round6_depth_codec.py', 'code/experiments/bank.py', 'code/experiments/fused_progressive.py']
    primary = ['code/a2/round7_runner.py', 'code/a2/round6_runner.py', 'code/a2/run_representation_variants.py',
               'code/a2/factory.py', 'code/a2/session.py', 'code/a2/session_boundary.py',
               'code/a2/round6_depth_bank.py', 'code/a2/round6_depth_attention.py']
    key = ['code/a2/round7_key_bank.py', 'code/a2/round7_key_attention.py', 'code/a2/round7_key_codec.py',
           'code/a2/round7_key_layout.py', 'code/a2/round7_key_artifacts.py']
    second = ['code/a2/round7_second_packed.py', 'code/a2/round6_second_session.py',
              'code/a2/round7_tail_key_bank.py', 'code/a2/round7_tail_key_attention.py', 'code/a2/round7_tail_key_codec.py']
    narrow = common + (primary if methods - {'native_hf', 'full', 'round7_tail_key'} else [])
    if 'round7_key' in methods or 'round7_tail_key' in methods:
        narrow += key
    if 'round7_tail_key' in methods:
        narrow += second
    source_entries = []
    for path in sorted({local_path(value) for value in narrow} | {path.resolve() for path in args.inference_sources}):
        item = file_entry(path, 'narrow declared inference dependency source; presence does not itself prove execution')
        files[item['path']] = item
        source_entries.append(item)
    analyses, analyzed = [], set()
    for path in args.analyses:
        path = path / 'summary.json' if path.is_dir() else path
        summary = read(path)
        require(summary.get('status') == 'COMPLETE' and summary.get('audit', {}).get('status') == 'PASS',
                'Selected development analysis is not complete with passing audits')
        role = str(summary.get('analysis_role', ''))
        require(role and 'confirm' not in role.lower(), 'Use explicitly declared development analysis before confirmation')
        item = file_entry(path, 'completed development quality/owner analysis')
        files[item['path']] = item
        names = {row['variant'] for row in summary['table']}
        analyzed.update(names)
        analyses.append(dict(path=item['path'], sha256=item['sha256'], analysis_role=role,
            reported_variants=sorted(names), selected_variants_in_analysis=sorted(names & set(chosen))))
        for run in summary.get('audit', {}).get('runs', []):
            manifest_path = local_path(run['run']) / 'manifest.json'
            if manifest_path.is_file():
                manifest = read(manifest_path)
                require(sha(manifest_path) == run['manifest_sha256'], 'Development analysis manifest changed')
                for variant in manifest['config']['variants']:
                    if variant['name'] in chosen:
                        require(variant == chosen[variant['name']], 'Selected method differs from its analyzed configuration')
    for path, purpose in [(DOC_POOL, 'fixed answer-free new Doc128 identities'), (TEXT_POOL, 'fixed answer-free new Text128 identities'),
                          (ROOT / 'audit/a2/round7_development_selection_v1.json', 'source-only calibration/development split freeze'),
                          (ROOT / 'audit/a2/round7_development_membership_v1.json', 'calibration/development sample/reference fingerprints')]:
        item = file_entry(path, purpose)
        files[item['path']] = item
    # Source disjointness is checked against every train/select source, and the
    # exact original image identities remain part of the pinned pool files.
    membership = read(ROOT / 'audit/a2/round7_development_membership_v1.json')
    development_doc, development_text, hashes = set(), set(), set()
    for label, spec in membership['datasets'].items():
        if 'independent' not in label:
            continue
        sample_path = ROOT / spec['samples']
        require(sha(sample_path) == spec['samples_sha256'], 'Development source split changed')
        for row in read(sample_path):
            hashes.update(row['hashes'])
            if label.startswith('doc'):
                development_doc.add(str(row['source_document_id']))
            else:
                development_text.add(str(row['source_original_image_id']))
    doc_sources = {row['source_document_id'] for row in doc['candidates128']}
    text_sources = {row['source_image_id'] for row in text['images_manifest']}
    doc_hashes = {row['sha256'] for row in doc['candidates128']}
    text_hashes = {row['sha256'] for row in text['images_manifest']}
    require(not (development_doc & doc_sources or development_text & text_sources
                 or hashes & doc_hashes or hashes & text_hashes or doc_hashes & text_hashes), 'Development/confirmation source or content overlap')
    result = dict(schema='trajkv.round7.method-freeze.v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        status='FROZEN_INPUT_IDENTITIES', selected_variants=chosen, frozen_files=list(files.values()),
        inference_source_narrowlist=source_entries,
        inference_source_scope='declared source dependency identities, separate from raw manifest execution-version receipts',
        development_analyses=analyses, selected_method_has_completed_development_analysis={name: name in analyzed for name in chosen},
        confirmation_references_materialized_at_creation=False,
        cohorts=dict(doc=dict(sources=128, questions=doc['proposed_question_count'], max_questions_per_source=4,
                             original_split='train', pool_sha256=sha(DOC_POOL)),
                     text=dict(sources=128, questions=128, pool_sha256=sha(TEXT_POOL))),
        development_confirmation_source_content_overlap=0, GPU_used=False,
        source_sha256=sha(__file__), note=args.note,
        scope='freeze of concrete methods, calibration inputs, analysis receipts and new source pools; does not establish novelty, superiority or backbone-pretraining exclusion')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    verify(args.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    p.add_argument('--configs', nargs='+', type=Path, default=[])
    p.add_argument('--analyses', nargs='*', type=Path, default=[])
    p.add_argument('--artifacts', nargs='*', type=Path, default=[])
    p.add_argument('--inference-sources', nargs='*', type=Path, default=[])
    p.add_argument('--selected-variants', nargs='*', default=[])
    p.add_argument('--note', default='')
    args = p.parse_args()
    verify(args.verify) if args.verify else create(args)
