"""Explicit primary plus actual follow-up publication scope; no execution/publish.

V1 remains the primary28/1120 contract. V3 separately seals middle, bounded
KRA development and conditional reserve224 replication. Reused reference
cells point to primary artifacts and never add fresh inference events.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib
from pathlib import Path
import re
import sys
from round10_public_contract_v1 import (ROOT, EVIDENCE, FINAL_STATES, file_allowed,
    require, sha, load, canonical, inside, write_new, validate as validate_primary)

sys.dont_write_bytecode = True
SCHEMA = 'trajkv.round10.public_stage_contract.v3'
SEAL_SCHEMA = 'trajkv.round10.public_stage_seal.v3'
DOMAINS = ('textvqa', 'infovqa', 'docvqa', 'chartqa')
REFS = ('FP16', 'R7low', 'R7high')
MIDDLE = ('V128-R43', 'AddOldOrder')
MIDDLE_COUNTS = dict(textvqa=64, infovqa=64, docvqa=16, chartqa=16)
EXTENSION_COUNTS = dict(textvqa=64, infovqa=64, docvqa=48, chartqa=48)
CANDIDATES = ('LS-FZ', 'LS-RE', 'FB0', 'FB-W', *MIDDLE, 'KRA1')
ADAPTERS = {'middle_development': 'round10_middle_harvest_v2',
            'extension_retest': 'round10_extension_harvest_v1', 'kra_development': 'round10_kra_harvest_v1'}
ADAPTER_PINS = {'middle_development': 'a26076dcaa0d1d9e6073042a11cc8ccb7629990b5b1e8a99432a75aa9fc80aac',
                'extension_retest': '2d5e77935b36dc529e4da337046556cba0eb20618ee7ef6fd9eeef2d84dda5ac',
                'kra_development': '4014a59966f773d848cfa9e00550b02cf286e89123cc98542cfa9e522c5e1096'}
# New KRA/extension epochs are intentionally not guessed before author seal.
KRA_EXTENSION_ADAPTER = 'round10_extension_harvest_v2'
KRA_EXTENSION_ADAPTER_SHA = None
MIDDLE_FROZEN_HELPER_SHA = '4bd78e263c23680f2162efdb1a0277b3029bdb70a53e2bd60038c2b10b191ebe'


def candidate():
    middle_rows = [{'dataset': d, 'mode': m, 'expected_QA': MIDDLE_COUNTS[d],
                    'disposition': 'PENDING', 'reuse_from':
                    {'group_id': 'primary', 'dataset': d, 'mode': m} if m in REFS else None}
                   for d in DOMAINS for m in (*REFS, *MIDDLE)]
    return {'schema': SCHEMA, 'phase_label': 'ROUND10_COMPLETE_EXECUTED_SCOPE',
        'stage_state': 'PENDING_ROOT_SCOPE_SEAL', 'root_seal': None,
        'primary_contract': None, 'whole_Round10_goal_complete_claimed': False,
        'SourceRun_admission': False,
        'kra_calibration': {'disposition': 'PENDING', 'raw': None, 'independent_source_proof': None,
                            'public_scalar_and_source_files': []},
        'groups': [{'group_id': 'middle', 'kind': 'middle_development',
            'execution_disposition': 'PENDING', 'cohort_label': 'D160 development; not independent confirmation',
            'counts': MIDDLE_COUNTS, 'candidate_modes': list(MIDDLE),
            'plans': [], 'harvest_root': None, 'refs_root': None, 'cells': middle_rows},
            {'group_id': 'kra', 'kind': 'kra_development', 'execution_disposition': 'PENDING',
             'cohort_label': 'D160 bounded KRA1 development; not independent confirmation',
             'counts': MIDDLE_COUNTS, 'candidate_modes': ['KRA1'], 'plans': [], 'harvest_root': None, 'refs_root': None,
             'cells': [{'dataset': d, 'mode': m, 'expected_QA': MIDDLE_COUNTS[d], 'disposition': 'PENDING',
                        'reuse_from': {'group_id': 'primary', 'dataset': d, 'mode': m} if m in REFS else None}
                       for d in DOMAINS for m in (*REFS, 'KRA1')]},
            {'group_id': 'extension', 'kind': 'extension_retest',
            'execution_disposition': 'PENDING',
            'cohort_label': 'reserved unused224 replication; same selected methods, no parameter reselection',
            'counts': EXTENSION_COUNTS, 'candidate_modes': [],
            'plans': [], 'harvest_root': None, 'refs_root': None, 'cells': []}],
        'files': [], 'no_private_tensors_images_weights_parquet_chat_or_credentials': True,
        'history_and_all_previous_namespaces_releases_must_be_preserved': True}


def seal_payload(c):
    """Root writes this only after choosing every actual final disposition."""
    return {'schema': SEAL_SCHEMA, 'status': 'SEALED_FOR_LOCAL_PUBLIC_PACKAGE',
        'phase_label': c['phase_label'], 'primary_contract_sha256': c['primary_contract']['sha256'],
        'groups_sha256': canonical(c['groups']), 'files_sha256': canonical(c['files']),
        'kra_calibration_sha256': canonical(c['kra_calibration']),
        'whole_Round10_goal_complete_claimed': False,
        'history_and_all_previous_namespaces_releases_must_be_preserved': True}


def readiness(c):
    require(c.get('schema') == SCHEMA and re.fullmatch('[A-Za-z0-9_.-]+', c['phase_label']), 'explicit v3 scope required')
    require(c.get('whole_Round10_goal_complete_claimed') is False and c.get('SourceRun_admission') is False,
            'local publication scope cannot close Goal or grant SourceRun admission')
    require({g['kind'] for g in c['groups']} == set(ADAPTERS) and
            len(c['groups']) == 3 and len({g['group_id'] for g in c['groups']}) == 3,
            'explicit middle, KRA and extension dispositions required')
    missing = []
    if c.get('stage_state') != 'ROOT_SEALED': missing.append('Root v3 scope seal')
    if not c.get('root_seal'): missing.append('root_seal path/SHA')
    if not c.get('primary_contract'): missing.append('primary v1 sealed contract path/SHA')
    if c['kra_calibration']['disposition'] not in ('CLOSED_COMPLETE', 'STOPPED', 'NOT_RUN'):
        missing.append('KRA calibration actual final disposition')
    for g in c['groups']:
        if g['execution_disposition'] not in ('EXECUTED', 'NOT_RUN'):
            missing.append(g['group_id'] + ': actual execution disposition')
        if g['execution_disposition'] == 'EXECUTED':
            pending = KRA_EXTENSION_ADAPTER_SHA if g['kind'] == 'extension_retest' and 'KRA1' in g['candidate_modes'] else ADAPTER_PINS[g['kind']]
            if pending is None: missing.append(g['group_id'] + ': final actual source adapter epoch not sealed')
            for r in g['cells']:
                if r['disposition'] not in FINAL_STATES:
                    missing.append(g['group_id'] + '/' + r['dataset'] + '/' + r['mode'] + ': final disposition')
    return {'status': 'WAIT_ROUND10_V3_ROOT_SCOPE_SEAL' if missing else 'READY_FOR_DEEP_LOCAL_VALIDATION',
            'missing': missing, 'publication_eligible': False, 'whole_Round10_goal_complete_claimed': False,
            'GitHub_mutation': False, 'new_GPU_jobs_started': 0}


def add_file(files, path, role, expected=None, public_path=None, aliases=()):
    path = Path(path).resolve()
    require((inside(path, ROOT) or inside(path, EVIDENCE)) and path.is_file(), 'missing/outside explicit source: ' + str(path))
    file_allowed(path, role)
    digest = sha(path); require(expected is None or digest == expected, 'source SHA differs: ' + str(path))
    origin = ROOT if inside(path, ROOT) else EVIDENCE
    rel = public_path or path.relative_to(origin).as_posix()
    require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and '\\' not in rel, 'unsafe public path')
    st = path.stat(); require(path.suffix == '.jsonl' or st.st_size <= 64*1024**2, 'large non-JSONL source requires review')
    if rel in files:
        require(sha(files[rel]['path']) == digest, 'public source collision: ' + rel)
        files[rel]['aliases'] = list(dict.fromkeys([*files[rel].get('aliases', []), str(path), *aliases]))
    else:
        files[rel] = {'path': path, 'role': role, 'origin': 'project' if origin == ROOT else 'evidence_mirror',
            'source_path': path.relative_to(origin).as_posix(), 'stat': (st.st_size, st.st_mtime_ns, st.st_ctime_ns),
            'large': path.suffix == '.jsonl', 'aliases': list(aliases)}
    return rel


def validate(contract_path, deep=False):
    c = load(contract_path); ready = readiness(c)
    if ready['missing']: return ready, {}, c
    sealref = c['root_seal']; require(sha(sealref['path']) == sealref['sha256'], 'v3 root seal changed')
    require(load(sealref['path']) == seal_payload(c), 'Root v3 seal does not bind actual full scope')
    require(c['history_and_all_previous_namespaces_releases_must_be_preserved'] is True and
            c['no_private_tensors_images_weights_parquet_chat_or_credentials'] is True, 'history/privacy declaration required')
    ref = c['primary_contract']; require(sha(ref['path']) == ref['sha256'], 'primary contract changed')
    phase, files, primary = validate_primary(ref['path'], deep=deep)
    if not phase['publication_eligible']:
        return {**ready, 'status': 'WAIT_PRIMARY_V1_SEAL_OR_RESULTS', 'missing': phase['missing']}, {}, c
    add = lambda *a, **kw: add_file(files, *a, **kw)
    add(contract_path, 'sealed_followup_scope_contract', public_path='evidence/ROUND10_SUPPLEMENT_CONTRACT.json')
    add(sealref['path'], 'root_followup_scope_seal', sealref['sha256'], 'evidence/ROUND10_SUPPLEMENT_SEAL.json')
    for f in c['files']: add(f['path'], f['role'], f['sha256'], f.get('public_path'), f.get('source_aliases', []))
    prior = {('primary', r['dataset'], r['mode']): r for r in primary['cells']}
    outputs = []
    for g in c['groups']:
        label, kind = g['group_id'], g['kind']
        expected_counts = EXTENSION_COUNTS if kind == 'extension_retest' else MIDDLE_COUNTS
        require(g['counts'] == expected_counts and isinstance(g['cohort_label'], str) and g['cohort_label'], 'followup cohort/count label differs')
        if g['execution_disposition'] == 'NOT_RUN':
            require(not g['cells'] and not g['plans'] and g.get('reason') and g.get('disposition_evidence'), 'unrun followup needs reason/evidence and no invented runs')
            e = g['disposition_evidence']; add(e['path'], 'followup_nonexecution_disposition', e['sha256'])
            outputs.append({'group_id': label, 'kind': kind, 'execution_disposition': 'NOT_RUN',
                'view_cells': 0, 'closed_quality_answer_events': 0, 'new_answer_events': 0, 'reused_answer_events': 0, 'cells': []})
            continue
        modes = tuple(g['candidate_modes'])
        require((modes == MIDDLE if kind == 'middle_development' else modes == ('KRA1',) if kind == 'kra_development' else
                 len(modes) in (1, 2) and len(set(modes)) == len(modes) and set(modes) <= set(CANDIDATES)), 'wrong followup selected modes')
        all_modes = (*REFS, *modes); rows = g['cells']
        require(len(rows) == 4*len(all_modes) and {(r['dataset'], r['mode']) for r in rows} ==
                {(d, m) for d in DOMAINS for m in all_modes}, 'followup exact declared matrix differs')
        require(all(r['expected_QA'] == expected_counts[r['dataset']] and r['disposition'] in FINAL_STATES for r in rows), 'invalid followup cell disposition')
        adapter_name = KRA_EXTENSION_ADAPTER if kind == 'extension_retest' and 'KRA1' in modes else ADAPTERS[kind]
        adapter_digest = KRA_EXTENSION_ADAPTER_SHA if adapter_name == KRA_EXTENSION_ADAPTER else ADAPTER_PINS[kind]
        adapter = importlib.import_module(adapter_name); adapter.check_pins()
        adapter_path = ROOT/'ops'/(adapter_name+'.py')
        add(adapter_path, 'executed_followup_harvest_source', adapter_digest)
        add(ROOT/'ops/round10_middle_harvest_v1.py', 'frozen_predecessor_harvest_source', MIDDLE_FROZEN_HELPER_SHA)
        for p in g['plans']: add(p['path'], 'followup_execution_plan', p['sha256'])
        loaded = adapter.load_plans([Path(p['path']) for p in g['plans']]); items, origins = loaded[:2]
        declared = {(r['dataset'], r['mode']) for r in rows}
        require(set(items) <= declared, 'execution plans contain undeclared followup cells')
        if kind == 'extension_retest':
            require(len(loaded) == 3 and tuple(loaded[2]) == modes and set(items) == declared,
                    'actual extension plans/selected modes differ from Root sealed scope')
        else:
            require({(d, m) for d in DOMAINS for m in modes} <= set(items),
                    'development plans must declare all actual new candidate cells')
        require(g['refs_root'] and g['harvest_root'], 'actual followup reference/harvest roots required')
        rr = Path(g['refs_root'])
        for name in ('source_lock.json',): add(rr/name, 'followup_source_lock')
        group_rows = []; fresh = reused = closed_events = 0
        for r in rows:
            ds, mode, state, n = r['dataset'], r['mode'], r['disposition'], r['expected_QA']
            reuse = r.get('reuse_from')
            if kind in ('middle_development', 'kra_development') and mode in REFS:
                require(reuse == {'group_id': 'primary', 'dataset': ds, 'mode': mode}, 'middle references must explicitly reuse primary')
                src = prior['primary', ds, mode]
                require(state == src['disposition'] and n == src['expected_QA'], 'reused disposition/count differs')
                ref = next((x for x in phase['closed'] + phase['failures'] if (x['dataset'], x['mode']) == (ds, mode)), None)
                require(ref is not None, 'reused reference must have actual primary raw/score artifacts')
                group_rows.append({**r, 'raw': ref['raw'], 'scored': ref['scored'], 'primary_mean': ref['primary_mean']})
                reused += n
                if state == 'CLOSED_COMPLETE': closed_events += n
                continue
            require(reuse is None, 'only middle reference cells may reuse primary inference')
            if state in ('STOPPED', 'NOT_RUN'):
                e = r.get('disposition_evidence'); require(e and r.get('reason') and r.get('primary_mean') is None, 'unrun cell requires evidence and null quality')
                add(e['path'], 'followup_cell_noncompletion_disposition', e['sha256'])
                group_rows.append({**r, 'primary_mean': None}); continue
            require((ds, mode) in items, 'executed followup lacks actual plan item')
            item = items[ds, mode]; name = item['cell']['name']
            raw = Path(g['harvest_root'])/'raw'/name; scored = Path(g['harvest_root'])/'scored'/name
            m, done, cfg, proof = adapter.check_local_proof(raw, item)
            require(done['expected_QA'] == done['recorded_QA'] == n, 'followup raw count differs')
            require((done['status'] == 'PASS_RAW_COMPLETE') == (state == 'CLOSED_COMPLETE'), 'followup raw disposition differs')
            if state == 'CLOSED_COMPLETE': require(proof['parent_cell_success'] is True, 'successful followup needs committed parent exit0')
            for e in proof['files']: add(raw/e['file'], 'executed_followup_source_or_raw', e['sha256'], aliases=[e['remote_file']])
            for name in ('source_retrieval_receipt_round10_v1.json', 'retrieval_receipt.json'):
                add(raw/name, 'independent_followup_source_proof')
            require((scored/'summary.json').is_file(), 'no existing followup score; export cannot invent one')
            s = load(scored/'summary.json')
            require(s['expected_run_questions'] == n and (s['primary_mean'] is not None) == (state == 'CLOSED_COMPLETE'), 'followup complete/failure score differs')
            for name in ('summary.json', *s['derived_file_hashes']):
                add(scored/name, 'frozen_followup_offline_score', None if name == 'summary.json' else s['derived_file_hashes'][name])
            inputname = Path(cfg['inputs']).name
            for name, role in ((inputname, 'followup_source_inputs'), (inputname.replace('_inputs.jsonl', '_refs.jsonl'), 'followup_reference_answers')):
                add(rr/name, role)
            if deep:
                point, _ = adapter.closed_cell(raw, scored, item, rr)
                require((point['primary_mean'] is not None) == (state == 'CLOSED_COMPLETE'), 'followup source/physical/score verification differs')
            group_rows.append({**r, 'raw': str(raw), 'scored': str(scored), 'primary_mean': s['primary_mean'], 'config_sha256': m['config_sha256']})
            fresh += n
            if state == 'CLOSED_COMPLETE': closed_events += n
        outputs.append({'group_id': label, 'kind': kind, 'execution_disposition': 'EXECUTED',
            'cohort_label': g['cohort_label'], 'view_cells': len(rows), 'declared_view_answer_events': sum(r['expected_QA'] for r in rows),
            'harvest_adapter': {'file': 'ops/' + adapter_name + '.py', 'source_sha256': adapter_digest},
            'new_answer_events': fresh, 'reused_answer_events': reused, 'closed_quality_answer_events': closed_events,
            'plans': origins, 'cells': group_rows})
    calibration = c['kra_calibration']
    if calibration['disposition'] == 'CLOSED_COMPLETE':
        cal = Path(calibration['raw']); proof = calibration['independent_source_proof']
        require(proof and sha(proof['path']) == proof['sha256'], 'independent KRA calibration proof required')
        add(proof['path'], 'kra_calibration_independent_source_proof', proof['sha256'])
        add(ROOT/'ops/round10_kra_fetch_calibration_v1.py', 'independent_kra_calibration_retrieval_source',
            '4aff5ffc3f81172faab35226f46c5d7a95f589da06e65cc724408c6e6010f566')
        proofdoc = load(proof['path'])
        require(proofdoc.get('remote_hashes_computed_independently') is True or proofdoc.get('independent_remote_SHA') is True,
                'KRA calibration proof is not independent remote readback')
        require(isinstance(proofdoc.get('files'), list) and proofdoc['files'], 'KRA calibration actual source/raw file proof required')
        for e in proofdoc['files']:
            rel = Path(e['file']); require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe KRA calibration proof member')
            add(cal/rel, 'kra_calibration_source_or_raw', e['sha256'], aliases=[e['remote_file']])
        for f in calibration['public_scalar_and_source_files']:
            add(f['path'], 'kra_calibration_scalar_or_source', f['sha256'], f.get('public_path'), f.get('source_aliases', []))
        required = ('manifest.json', 'records.jsonl', 'complete.json', 'metric_fit48.json', 'selected_metric.json', 'selection.json', 'identity.json')
        require(all(any(v['path'] == (cal/n).resolve() for v in files.values()) for n in required), 'KRA calibration scalar/source closure incomplete')
        complete = load(cal/'complete.json')
        require(complete['status'] == 'PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16' and complete['actual_GPU'] is True and
                complete['fit_sources_reused'] == 48 and complete['new_fit_source_captures'] == 0 and complete['select_sources'] == 16,
                'KRA identity-only or preparation cannot stand in for select16 completion')
        for n, digest in {'records.jsonl': complete['records_sha256'], 'selected_metric.json': complete['metric_sha256'],
                          'selection.json': complete['selection_sha256'], 'identity.json': complete['identity_sha256']}.items():
            require(sha(cal/n) == digest, 'KRA calibration scalar identity differs')
    else:
        e = calibration.get('disposition_evidence'); require(e and calibration.get('reason'), 'stopped/unrun KRA calibration needs evidence')
        add(e['path'], 'kra_calibration_noncompletion_disposition', e['sha256'])
    phase = {**phase, 'supplement_scope': outputs, 'kra_calibration_disposition': calibration['disposition'],
        'whole_Round10_goal_complete_claimed': False, 'v3_root_sealed': True,
        'primary_scope_only_cells': 28, 'primary_scope_only_planned_events': 1120,
        'primary_actual_answer_events': sum(r['expected_QA'] for r in phase['closed'] + phase['failures']),
        'total_declared_view_answer_events': 1120 + sum(g.get('declared_view_answer_events', 0) for g in outputs),
        'total_actual_view_answer_events': sum(r['expected_QA'] for r in phase['closed'] + phase['failures']) +
            sum(g['new_answer_events'] + g['reused_answer_events'] for g in outputs),
        'total_actual_fresh_answer_events': sum(r['expected_QA'] for r in phase['closed'] + phase['failures']) +
            sum(g['new_answer_events'] for g in outputs),
        'followup_new_answer_events': sum(g['new_answer_events'] for g in outputs),
        'followup_reused_answer_events': sum(g['reused_answer_events'] for g in outputs)}
    return phase, files, c


def main():
    p = argparse.ArgumentParser(); p.add_argument('--contract', type=Path); p.add_argument('--candidate-output', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--deep', action='store_true'); a = p.parse_args()
    require(inside(a.output, EVIDENCE) and not a.output.exists(), 'new E report output required')
    if a.candidate_output:
        require(a.contract is None and inside(a.candidate_output, EVIDENCE), 'candidate only on E')
        write_new(a.candidate_output, candidate()); a.contract = a.candidate_output
    require(a.contract is not None, 'contract required'); report, _, _ = validate(a.contract, a.deep)
    write_new(a.output, report); print(__import__('json').dumps(report, ensure_ascii=False)); return 0 if report['publication_eligible'] else 3


if __name__ == '__main__': raise SystemExit(main())
