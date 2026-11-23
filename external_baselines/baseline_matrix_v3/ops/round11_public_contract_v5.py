"""Explicit Round11 cutoff and portable source inventory. Never launches work.

Root alone supplies a final seal. Groups follow actual pinned harvest plans;
unexecuted branches retain null quality and hashed disposition evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('${EVIDENCE_ROOT}').resolve()
SCHEMA = 'trajkv.round11.public_stage_contract.v5'
SUPPORTED_SCHEMAS = (SCHEMA, 'trajkv.round11.public_stage_contract.v4', 'trajkv.round11.public_stage_contract.v3')
SEAL_SCHEMA = 'trajkv.round11.public_stage_seal.v5'
BASE_COMMIT = '43d03af0e8938aeb5f73ffc0c104093a7bd14d73'
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')
FINAL_STATES = ('CLOSED_COMPLETE', 'CLOSED_WITH_FAILURES', 'STOPPED', 'NOT_RUN')
SAFE_SUFFIXES = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.toml', '.yaml', '.yml'}
REQUIRED_ROLES = {'scope_report', 'quality_bytes_table'}
PORTABLE_SOURCES = ('code/score.py', 'code/metrics.py', 'code/vendor_textvqa_metric.py',
    'code/compare_runs.py', 'ops/round10_harvest_score_v1.py',
    'ops/prepare_remaining_phase_delivery_v1.py', 'ops/verify_public_round10_v3.py',
    'ops/round11_public_confirmation_v5.py', 'ops/round11_public_aliases_v5.py', 'ops/round11_public_metric_contracts_v5.py', 'ops/round11_public_expansion_v5.py')


def require(value, message):
    if not value: raise ValueError('ROUND11_PUBLIC: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
def inside(path, root): return Path(path).resolve().is_relative_to(Path(root).resolve())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def candidate():
    return {'schema': SCHEMA, 'phase_label': 'ROUND11_ACTUAL_ROOT_CUTOFF',
        'stage_state': 'PENDING_ROOT_SCOPE_SEAL', 'root_seal': None,
        'SourceRun_admission': False, 'whole_external_library_completion_claimed': False,
        'base_public_commit': BASE_COMMIT,
        'history_and_all_previous_namespaces_releases_must_be_preserved': True,
        'source_public_Git_LF_archive_identities_separate': True,
        'no_private_tensors_images_weights_parquet_chat_or_credentials': True,
        'packager_transport_dependencies_publicly_rerunnable': False,
        'groups': [{'group_id': name, 'kind': kind, 'execution_disposition': 'PENDING',
            'cohort_label': label, 'plan': None, 'adapter': None, 'counts': {},
            'modes': [], 'harvest_root': None, 'summary_root': None, 'cells': [],
            'paired_contrasts': []} for name, kind, label in (
                ('primary', 'development', 'SourceGroupD96 development'),
                ('wo', 'bounded_development', 'Conditional W_O development; actual plan required'),
                ('confirm128', 'fresh_confirmation', 'Original unused R128; frozen selection before original outcomes'),
                ('expand384', 'fresh_fixed_extension', 'Additional384 fixed after partial original128 observation; methods frozen; original/additional/combined reports separate'))],
        'supporting_evidence': [], 'files': []}


def seal_payload(c):
    return {'schema': 'trajkv.round11.public_stage_seal.'+c['schema'].rsplit('.',1)[1], 'status': 'SEALED_FOR_LOCAL_PUBLIC_PACKAGE',
        'phase_label': c['phase_label'], 'groups_sha256': canonical(c['groups']),
        'files_sha256': canonical(c['files']), 'supporting_evidence_sha256': canonical(c['supporting_evidence']),
        'base_public_commit': c['base_public_commit'], 'SourceRun_admission': False,
        'whole_external_library_completion_claimed': False,
        'history_and_all_previous_namespaces_releases_must_be_preserved': True}


def readiness(c):
    require(c.get('schema') in SUPPORTED_SCHEMAS and re.fullmatch('[A-Za-z0-9_.-]+', c.get('phase_label', '')), 'explicit stage schema/label required')
    require(c.get('SourceRun_admission') is False and c.get('whole_external_library_completion_claimed') is False, 'custom stage cannot establish canonical admission or library completion')
    require(c.get('base_public_commit') == BASE_COMMIT, 'Round10 preserved history anchor differs')
    require(isinstance(c.get('groups'), list) and c['groups'] and len({g['group_id'] for g in c['groups']}) == len(c['groups']), 'unique actual scope groups required')
    missing = []
    if c.get('stage_state') != 'ROOT_SEALED': missing.append('Root actual final cutoff seal')
    if not c.get('root_seal'): missing.append('Root seal path/SHA')
    for group in c['groups']:
        require(re.fullmatch('[A-Za-z0-9_.-]+', group['group_id']), 'unsafe group id')
        if group.get('execution_disposition') not in ('EXECUTED', 'NOT_RUN'): missing.append(group['group_id'] + ': final execution disposition')
        if group.get('execution_disposition') == 'EXECUTED':
            for field in ('plan', 'adapter', 'harvest_root', 'summary_root'):
                if not group.get(field): missing.append(group['group_id'] + ': ' + field)
            for row in group.get('cells', []):
                if row.get('disposition') not in FINAL_STATES: missing.append(group['group_id'] + ': final cell disposition')
    for item in c.get('supporting_evidence', []):
        if item.get('disposition') not in FINAL_STATES: missing.append(item['evidence_id'] + ': final supporting disposition')
    return {'status': 'WAIT_ROUND11_ROOT_SEAL_AND_ACTUAL_SCOPE' if missing else 'READY_FOR_DEEP_LOCAL_VALIDATION',
        'missing': missing, 'publication_eligible': False, 'GitHub_mutation': False, 'new_GPU_jobs_started': 0,
        'SourceRun_admission': False, 'quality_QA_inferred_from_plans': 0}


def file_allowed(path, role):
    p = Path(path); name = p.name.lower()
    require(p.suffix.lower() in SAFE_SUFFIXES or name.startswith(('license', 'copying', 'notice')) or (p.suffix.lower() == '.log' and role == 'frozen_data_job_log'), 'binary/image/model/tensor/parquet/archive excluded')
    require(not {x.lower() for x in p.parts} & {'.git', '__pycache__', 'models', 'weights', 'images', 'cache', 'tmp', 'private_fit_shards', 'private_select_teachers'}, 'private/cache directory excluded')
    require(not any(x in name for x in ('chat', '私聊', '对话记录', 'doubao', 'gpt40_reply', 'raw_logits', 'raw_tf_logits')) and not any('private' in x.lower() for x in p.parts), 'private discussion/tensor/logits area excluded')
    require(isinstance(role, str) and role and not role.startswith('private_'), 'explicit public evidence role required')


def add_file(files, path, role, expected=None, public_path=None, aliases=()):
    supplied = Path(path).absolute()
    path = supplied.resolve()
    require((inside(path, ROOT) or inside(path, EVIDENCE)) and path.is_file(), 'missing/outside explicit source: ' + str(path))
    file_allowed(path, role); digest = sha(path)
    if path.suffix.lower() == '.log': require(role == 'frozen_data_job_log' and public_path is not None and Path(public_path).suffix.lower() == '.txt', 'only explicit source-verified data-job logs represented as public text are allowed')
    verified = []
    if str(supplied) != str(path):
        from round11_public_aliases_v5 import verify_local_alias
        file_allowed(supplied, role)
        verified.append(verify_local_alias(supplied, path, digest))
    require(expected is None or digest == expected, 'source SHA differs: ' + str(path))
    origin = ROOT if inside(path, ROOT) else EVIDENCE
    rel = public_path or path.relative_to(origin).as_posix(); pure = PurePosixPath(rel)
    require(not pure.is_absolute() and '..' not in pure.parts and '\\' not in rel and ':' not in rel, 'unsafe public path')
    st = path.stat(); require(path.suffix == '.jsonl' or st.st_size <= 64*1024**2, 'large non-JSONL source needs review')
    if rel in files:
        require(sha(files[rel]['path']) == digest, 'public source collision: ' + rel)
        files[rel]['aliases'] = list(dict.fromkeys([*files[rel].get('aliases', []), str(path), *aliases]))
        from round11_public_aliases_v5 import append_local_alias
        for record in verified: append_local_alias(files[rel].setdefault('verified_local_aliases', []), record)
    else:
        files[rel] = {'path': path, 'role': role, 'origin': 'project' if origin == ROOT else 'evidence_mirror',
            'source_path': path.relative_to(origin).as_posix(), 'stat': (st.st_size, st.st_mtime_ns, st.st_ctime_ns),
            'large': path.suffix == '.jsonl', 'aliases': list(aliases), 'verified_local_aliases': verified}
    return rel


def adapter_module(ref):
    path = Path(ref['path']).resolve()
    require(inside(path, ROOT/'ops') and sha(path) == ref['sha256'], 'explicit reviewed adapter source epoch differs')
    sys.path[:0] = [str(ROOT/'ops'), str(ROOT/'code')]
    spec = importlib.util.spec_from_file_location('_round11_contract_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if path.name == 'round11_confirmation_harvest_v2.py':
        from round11_public_confirmation_v5 import PINS, ADAPTER
        require(ref['sha256'] == PINS[ADAPTER] and all(hasattr(module, name) for name in ('load_plan', 'check_source', 'score_cell', 'shared')), 'exact frozen confirmation adapter API required')
    elif path.name == 'round11_expand_harvest_v1.py':
        from round11_public_expansion_v5 import PINS, ADAPTER
        require(PINS[ADAPTER] is not None and ref['sha256'] == PINS[ADAPTER] and all(hasattr(module, name) for name in ('load_plan', 'check_source', 'score_cell', 'shared')), 'exact frozen additional384 adapter API required')
    else:
        require(all(hasattr(module, name) for name in ('load_plan', 'selected_view', 'score_view', 'paired')), 'adapter must expose pinned CPU plan/view/score/paired interface')
    return module


def compare_json(expected, actual, message):
    require(canonical(expected) == canonical(actual), message)


def validate(contract_path, deep=False):
    c = load(contract_path); ready = readiness(c)
    if ready['missing']: return ready, {}, c
    require(c.get('history_and_all_previous_namespaces_releases_must_be_preserved') is True and c.get('no_private_tensors_images_weights_parquet_chat_or_credentials') is True and c.get('source_public_Git_LF_archive_identities_separate') is True and c.get('packager_transport_dependencies_publicly_rerunnable') is False, 'history/privacy/identity declarations required')
    ref = c['root_seal']; require(sha(ref['path']) == ref['sha256'], 'Root seal bytes changed')
    compare_json(load(ref['path']), seal_payload(c), 'Root seal does not bind the actual complete cutoff')
    files = {}; add = lambda *a, **kw: add_file(files, *a, **kw)
    add(contract_path, 'sealed_stage_contract', public_path='evidence/ROUND11_STAGE_CONTRACT.json')
    add(ref['path'], 'root_stage_scope_seal', ref['sha256'], 'evidence/ROUND11_STAGE_SEAL.json')
    for f in c['files']: add(f['path'], f['role'], f['sha256'], f.get('public_path'), f.get('source_aliases', []))
    require(REQUIRED_ROLES <= {x['role'] for x in files.values()}, 'explicit final report and browsable quality/bytes table required')
    for name in PORTABLE_SOURCES: add(ROOT/name, 'frozen_offline_source')
    groups = []; total = Counter(); fresh_sources = set(); reused_sources = {}
    for g in c['groups']:
        label = g['group_id']
        require(g.get('cohort_label') and g.get('kind'), 'actual cohort/kind labels required')
        if g['execution_disposition'] == 'NOT_RUN':
            require(not g['cells'] and not g.get('plan') and not g.get('harvest_root') and not g.get('summary_root') and g.get('primary_mean') is None and g.get('reason'), 'unrun group cannot invent plan/results/quality')
            e = g['disposition_evidence']; add(e['path'], 'nonexecution_disposition', e['sha256'])
            groups.append({'group_id': label, 'kind': g['kind'], 'execution_disposition': 'NOT_RUN', 'primary_mean': None, 'view_cells': 0, 'actual_fresh_answer_events': 0, 'reused_reference_view_events': 0, 'actual_view_answer_events': 0, 'closed_quality_answer_events': 0, 'cells': []}); continue
        adapter = adapter_module(g['adapter']); add(g['adapter']['path'], 'executed_harvest_source', g['adapter']['sha256'])
        add(g['plan']['path'], 'actual_execution_plan', g['plan']['sha256'])
        plan, source_group = adapter.load_plan(g['plan']['path'])
        if plan['schema'] == 'trajkv.round11.expand384_harvest_plan.v1':
            from round11_public_expansion_v5 import collect_group
            result = collect_group(g, adapter, plan, source_group, add, deep, fresh_sources, ROOT, c['groups'])
            groups.append(result)
            total.update({key: result.get(key, 0) for key in ('actual_view_answer_events', 'actual_fresh_answer_events', 'reused_reference_view_events', 'closed_quality_answer_events', 'view_cells', 'paired_comparisons', 'fresh_reference_answer_events', 'fresh_candidate_answer_events', 'fresh_successful_generation_QA', 'fresh_failure_status_QA')})
            continue
        if plan['schema'] == 'trajkv.round11.confirmation_harvest_plan.v2':
            from round11_public_confirmation_v5 import collect_group
            result = collect_group(g, adapter, plan, source_group, add, deep, fresh_sources, ROOT, c['groups'])
            groups.append(result)
            total.update({key: result.get(key, 0) for key in ('actual_view_answer_events', 'actual_fresh_answer_events', 'reused_reference_view_events', 'closed_quality_answer_events', 'view_cells', 'paired_comparisons', 'fresh_reference_answer_events', 'fresh_candidate_answer_events', 'fresh_successful_generation_QA', 'fresh_failure_status_QA')})
            continue
        from round11_public_confirmation_v5 import d96_runner_contract
        scripts = {e['item']['cell']['script'] for e in plan['entries'] if e['kind'] == 'fresh'}
        require(len(scripts) == 1, 'one explicit D96 candidate runner per group required')
        script = next(iter(scripts)); admission = d96_runner_contract(script)
        require(adapter.runner.SCHEMA == admission['raw_schema'] and list(adapter.runner.MODES) == admission['modes'] and adapter.PROOF_NAME == admission['proof_name'], 'D96 adapter declares the wrong actual runner/schema/proof namespace')
        add(plan['source_group_manifest_path'], 'source_group_manifest', plan['source_group_manifest_sha256'])
        add(source_group['reference_input_manifest_path'], 'input_manifest', plan['input_manifest_sha256'])
        counts = {d: source_group['datasets'][d]['count'] for d in DOMAINS}
        require(g['counts'] == counts and set(g['modes']) == {e['mode'] for e in plan['entries']} and len(g['modes']) == len(set(g['modes'])), 'actual group counts/modes differ from pinned plan')
        entries = {(e['dataset'], e['mode']): e for e in plan['entries']}
        rows = {(r['dataset'], r['mode']): r for r in g['cells']}
        require(len(rows) == len(g['cells']) and set(rows) == set(entries), 'every actual plan cell needs an exact final disposition')
        for domain in plan['domains'].values():
            for field, role in (('inputs', 'source_inputs'), ('refs', 'reference_answers'), ('source_lock', 'source_lock')):
                add(domain[field], role, domain[field+'_sha256'])
        summary_root = Path(g['summary_root']); summary = load(summary_root/'summary.json'); pairs = load(summary_root/'paired_comparisons.json')
        require(summary.get('plan_sha256') == g['plan']['sha256'], 'frozen summary refers to a different plan')
        for name in ('summary.json', 'paired_comparisons.json', 'quality_complete_bytes.csv'): add(summary_root/name, 'frozen_group_summary')
        frozen_points = {(p['dataset'], p['mode']): p for p in summary['points']}
        require(set(frozen_points) == set(entries) and len(frozen_points) == len(summary['points']) and summary['expected_cells_or_views'] == len(entries) and summary['expected_new_free_generation_QA'] == sum(counts[e['dataset']] for e in plan['entries'] if e['kind'] == 'fresh') and summary['expected_reused_reference_views'] == sum(counts[e['dataset']] for e in plan['entries'] if e['kind'] == 'reused') and summary['actual_closed_new_free_generation_QA'] == sum(p['fresh_free_generation_QA'] for p in summary['points']) and summary['actual_closed_reused_reference_views'] == sum(p['reused_reference_views'] for p in summary['points']), 'D96 summary actual matrix/fresh/reused accounting differs')
        outrows = []; bundles = {}; counter = Counter()
        for key, entry in entries.items():
            row = rows[key]; ds, mode = key; n = counts[ds]; state = row['disposition']
            require(row['expected_QA'] == n and row['kind'] == entry['kind'] and state in FINAL_STATES, 'cell count/kind/disposition differs')
            if state in ('STOPPED', 'NOT_RUN'):
                require(row.get('primary_mean') is None and row.get('reason'), 'stopped/unrun quality must remain null')
                require(frozen_points[key]['primary_mean'] is None, 'nonexecuted disposition conflicts with an existing primary score')
                e = row['disposition_evidence']; add(e['path'], 'cell_noncompletion_disposition', e['sha256'])
                outrows.append({**row, 'primary_mean': None}); continue
            raw = Path(g['harvest_root'])/'raw'/entry['item']['cell']['name'] if entry['kind'] == 'fresh' else Path(entry['source_raw_dir'])
            proofname = adapter.PROOF_NAME if entry['kind'] == 'fresh' else 'source_retrieval_receipt_round10_v1.json'
            proof = load(raw/proofname)
            if entry['kind'] == 'reused': require(sha(raw/proofname) == entry['source_receipt_sha256'], 'original reused producer receipt changed')
            for p in proof['files']: add(raw/p['file'], 'actual_producer_source_or_raw', p['sha256'], aliases=[p['remote_file']])
            for rel, digest in load(raw/'manifest.json')['source_hashes'].items():
                add(raw/'source_files'/rel, 'executed_runtime_source', digest, rel)
            for name in (proofname, 'retrieval_receipt.json'): add(raw/name, 'independent_actual_source_proof')
            view = Path(g['harvest_root'])/'views'/(ds+'__'+mode)
            require(all((view/name).is_file() for name in ('selected_records.jsonl', 'source_lineage.json', 'scored_records.jsonl', 'summary.json')), 'existing complete frozen view required; export never creates scores')
            for name in ('selected_records.jsonl', 'source_lineage.json', 'scored_records.jsonl', 'summary.json'): add(view/name, 'frozen_source_subset_score')
            point = load(view/'summary.json'); lineage = load(view/'source_lineage.json')
            for rel, digest in point['scoring_source_pins'].items():
                add(ROOT/rel, 'frozen_scoring_dependency', digest)
            add(ROOT/'audit/metrics_gate_v2.json', 'frozen_metric_gate')
            for rel, digest in load(ROOT/'audit/metrics_gate_v2.json')['code_hashes'].items():
                add(ROOT/rel, 'frozen_metric_gate_source', digest)
            require(point['expected_QA'] == n and point['canonical_SourceRun_admission_claimed'] is False and (point['primary_mean'] is not None) == (state == 'CLOSED_COMPLETE'), 'failure cannot become successful-subset primary')
            compare_json(point, frozen_points[key], 'group and per-view frozen score differ')
            require(lineage['fresh_inference_source'] is (entry['kind'] == 'fresh') and lineage['reused_reference_source'] is (entry['kind'] == 'reused') and lineage['inference_run'] is False, 'source subset is a view, not another inference run')
            if deep:
                selected = adapter.selected_view(entry, plan, source_group, g['harvest_root'])
                require(selected is not None, 'actual producer/source proof not closed')
                actual, bundle = adapter.score_view(entry, selected, plan)
                compare_json(point, actual, 'deep frozen quality/physical/source recomputation differs')
                if bundle is not None: bundles[key] = bundle
            counter['actual_view_answer_events'] += n
            counter['actual_fresh_answer_events' if entry['kind'] == 'fresh' else 'reused_reference_view_events'] += n
            if entry['kind'] == 'fresh':
                identity = (sha(raw/'manifest.json'), sha(raw/'records.jsonl'), sha(raw/'complete.json'))
                require(identity not in fresh_sources, 'fresh inference producer counted twice across groups'); fresh_sources.add(identity)
                counter['fresh_successful_generation_QA'] += point['fresh_free_generation_QA']; counter['fresh_failure_status_QA'] += n-point['fresh_free_generation_QA']
            else:
                identity = (sha(raw/'manifest.json'), sha(raw/'records.jsonl'), sha(raw/'complete.json'), tuple(lineage['ids']))
                require(identity not in reused_sources or reused_sources[identity] == n, 'reused source subset denominator differs'); reused_sources[identity] = n
            if state == 'CLOSED_COMPLETE': counter['closed_quality_answer_events'] += n
            outrows.append({**row, 'raw': str(raw), 'view': str(view), 'proof_name': proofname, 'primary_mean': point['primary_mean'], 'original_producer_QA': load(raw/'manifest.json')['expected_QA']})
        declared_pairs = {ds+'__'+candidate+'_minus_'+anchor for ds in DOMAINS for candidate, anchor in g['paired_contrasts'] if (ds, candidate) in bundles and (ds, anchor) in bundles} if deep else set(pairs)
        require(set(pairs) == declared_pairs, 'paired evidence coverage differs from declared actual closed contrasts')
        if deep:
            for name in declared_pairs:
                ds, pair = name.split('__', 1); cand, anchor = pair.split('_minus_', 1)
                compare_json(pairs[name], adapter.paired(bundles[ds, anchor], bundles[ds, cand]), 'paired CI/runtime/source recomputation differs')
        groups.append({'group_id': label, 'kind': g['kind'], 'cohort_label': g['cohort_label'], 'execution_disposition': 'EXECUTED', 'view_cells': len(rows), 'paired_comparisons': len(pairs), **dict(counter), 'cells': outrows})
        total.update(counter); total['view_cells'] += len(rows); total['paired_comparisons'] += len(pairs)
    for supporting in c['supporting_evidence']:
        require(supporting.get('task_quality_QA') == 0 and supporting.get('teacher_TF_or_technical_is_free_answer_QA') is False, 'QA0/teacher evidence cannot enter task quality')
        if supporting['disposition'] in ('STOPPED', 'NOT_RUN'):
            require(supporting.get('primary_mean') is None and supporting.get('reason'), 'noncompletion supporting evidence needs null quality/reason')
            e = supporting['disposition_evidence']; add(e['path'], 'supporting_noncompletion_disposition', e['sha256'])
        else:
            require(supporting.get('files'), 'actual public scalar/source supporting inventory required')
            for f in supporting['files']: add(f['path'], 'supporting_scalar_or_source', f['sha256'], f.get('public_path'), f.get('source_aliases', []))
            for assertion in supporting.get('scalar_assertions', []):
                require(any(Path(f['path']).resolve() == Path(assertion['path']).resolve() for f in supporting['files']), 'scalar assertion must name an explicit archived supporting file')
                value = load(assertion['path'])
                for key in assertion['keys']: value = value[key]
                require(value == assertion['value'], 'supporting scalar assertion differs')
    phase = {'status': 'PASS_ROUND11_ROOT_SEALED_LOCAL_STAGE', 'publication_eligible': bool(deep), 'deep_source_score_paired_physical_verified': bool(deep), 'phase_label': c['phase_label'], 'groups': groups, **dict(total), 'unique_reused_reference_view_events': sum(reused_sources.values()), 'repeated_reused_reference_view_events': total['reused_reference_view_events']-sum(reused_sources.values()), 'unique_reused_reference_producer_views': len(reused_sources), 'fresh_producer_cells': len(fresh_sources), 'recorded_QA_tasks_include_failure_status_rows': True, 'fresh_successful_generation_QA_is_reported_separately': True, 'SourceRun_admission': False, 'new_GPU_jobs_started': 0, 'GitHub_mutation': False, 'whole_external_library_completion_claimed': False}
    return phase, files, c


def main():
    p = argparse.ArgumentParser(); p.add_argument('--contract', type=Path); p.add_argument('--candidate-output', type=Path); p.add_argument('--deep', action='store_true'); a = p.parse_args()
    if a.candidate_output:
        require(a.contract is None and inside(a.candidate_output, EVIDENCE), 'candidate is local E preparation only'); write_new(a.candidate_output, candidate()); return 0
    require(a.contract is not None, '--contract required'); result, _, _ = validate(a.contract, a.deep); print(json.dumps(result, ensure_ascii=False)); return 0 if result['publication_eligible'] else 3


if __name__ == '__main__': raise SystemExit(main())
