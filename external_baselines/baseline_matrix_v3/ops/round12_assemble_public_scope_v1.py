"""Assemble explicit actual Round12 publication scope; never seal or publish.

Default requires every planned development/oracle/fixed500 producer closed.
--draft emits blockers and a non-eligible partial binding check only. Frozen
raw, datasets, metrics, launch proofs and earlier stages are never modified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
E = Path('${EVIDENCE_ROOT}')
BASE = E / 'results/round12'
DEVINPUT = E / 'results/round12_v1/bootstrap_inputs_v1'
REGISTRY = ROOT / 'configs/datasets/v3'
sys.path.insert(0, str(ROOT / 'code'))
from round12_public_core_v1 import LocalSource, sha, require, verify_semantics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def latest_state(folder):
    files = sorted((folder / 'collector_runs').glob('*/ROOT_STATE.json'))
    require(files, 'no readonly collector snapshot: ' + str(folder))
    return files[-1], read(files[-1])


class Assemble:
    def __init__(self, output):
        self.output = output
        self.files, self.cells, self.views, self.blockers = {}, {}, {}, []
        self.bundles = {}
        self.producers = {}
        captures = sorted((BASE / 'publication_producer_evidence_v1').glob('COLLECTION_*.json'))
        for path in captures:
            obj = read(path)
            if not isinstance(obj.get('jobs'), list):
                continue
            for job in obj['jobs']:
                name = job['name']
                score = (bool(job.get('QA0_result')), sum(bool(c.get('raw_manifest_closed')) for c in job.get('cells', [])),
                         bool(job.get('parent_complete')), len(job.get('cells', [])))
                if name not in self.producers or score >= self.producers[name][0]:
                    self.producers[name] = (score, job)

    def add(self, path, public, role, browser=True):
        path = Path(path)
        require(path.is_file(), 'actual selected file missing: ' + str(path))
        item = {'source_path': str(path), 'public_path': public, 'sha256': sha(path),
                'role': role, 'delivery_channel': 'BROWSER_AND_ARCHIVE' if browser else 'ARCHIVE_ONLY'}
        if public in self.files:
            require(self.files[public]['sha256'] == item['sha256'], 'public path would overwrite different evidence: ' + public)
        else:
            self.files[public] = item
        return public

    def entry(self, e):
        return self.add(e['source_path'], e['public_path'], e['role'],
                        PurePosixPath(e['public_path']).name not in ('records.jsonl', 'scored_records.jsonl'))

    def producer_fragment(self, name):
        matches = [(job, c) for _, job in self.producers.values() for c in job.get('cells', [])
                   if c.get('cell_id') == name and c.get('producer')]
        if not matches:
            self.blockers.append('actual producer source graph needs readonly refresh: '+name)
            return None
        require(len(matches) == 1, 'producer fragment absent/ambiguous: ' + name)
        job, fragment = matches[0]
        require(not fragment.get('unresolved'), 'source graph unresolved: ' + name)
        for key in ('launch', 'job', 'snapshot_inventory', 'entrypoint', 'job_config', 'parent_complete'):
            if job.get(key):
                self.entry(job[key])
        self.entry(fragment['source_config'])
        for e in fragment['files']:
            self.entry(e)
        if fragment['producer']['kind'] == 'sequence_child' and not job.get('parent_complete'):
            self.blockers.append('producer parent not closed/refreshed: ' + job['name'])
        return fragment

    def data_paths(self, manifest, *, refs=None):
        cfg = manifest['config']
        name = PurePosixPath(cfg['inputs']).name
        fixed = name.endswith('_p2_inputs.jsonl')
        folder = REGISTRY if fixed else DEVINPUT
        prefix = 'configs/data/round12_fixed500' if fixed else 'configs/data/round12_development'
        rname = name.replace('_inputs.jsonl', '_refs.jsonl')
        if refs is not None:
            rpath = Path(refs)
            rname = rpath.name
            name = rname.replace('_refs.jsonl', '_inputs.jsonl')
            folder = rpath.parent
        return {'inputs': self.add(folder / name, prefix + '/' + name, 'inputs'),
                'refs': self.add(folder / rname, prefix + '/' + rname, 'refs'),
                'source_lock': self.add(folder / 'source_lock.json', prefix + '/source_lock.json', 'source_lock')}

    def scored(self, directory, prefix):
        directory = Path(directory)
        result = {}
        for filename, key in (('summary.json', 'scored_summary'), ('scored_records.jsonl', 'scored_records')):
            result[key] = self.add(directory / filename, prefix + '/' + filename, 'scored', filename != 'scored_records.jsonl')
        for name in ('scores.csv', 'summary.csv'):
            if (directory / name).exists():
                self.add(directory / name, prefix + '/' + name, 'quality_table')
        return result

    def view(self, label, domain, cell_id, ids=None):
        key = (label, domain)
        v = self.views.setdefault(key, {'view_id': label + '_' + domain, 'label': label,
                                      'dataset': domain, 'members': []})
        member = {'cell_id': cell_id}
        if ids is not None:
            member['ids'] = ids
        if member not in v['members']:
            v['members'].append(member)

    def new_cell(self, item, *, oracle=False):
        name = item['name']
        item=dict(item)
        # A collector progress snapshot can precede a later immutable closed
        # retrieval. Admit that exact closed triplet and official score rather
        # than treating a mutable progress pointer as the evidence authority.
        folder=BASE/('oracle_harvest_v2' if oracle else ('fixed500_harvest_v1' if name.startswith('R_fixed500_') else 'harvest_v2'))
        rawdir,scored=folder/'raw'/name,folder/'scored'/name
        if (rawdir/'complete.json').is_file() and (rawdir/'retrieval_receipt.json').is_file():
            closed=read(rawdir/'complete.json')
            pilot=bool(item.get('pilot'))
            if closed.get('recorded_QA')==item['expected_QA'] and (pilot or (scored/'summary.json').is_file()):
                item.update(state='COMPLETE',local_raw=str(rawdir),scored_dir=str(scored))
        if item.get('state') != 'COMPLETE' or not item.get('local_raw'):
            self.blockers.append('raw/scoring not closed: ' + name)
            return
        raw = Path(item['local_raw'])
        m = read(raw / 'manifest.json')
        pilot = bool(item.get('pilot'))
        frag = self.producer_fragment(name)
        if frag is None:
            return
        prefix = 'results/round12/raw/' + name
        paths = {k: self.add(raw / filename, prefix + '/' + filename, 'raw', k != 'records')
                 for k, filename in (('manifest', 'manifest.json'), ('records', 'records.jsonl'), ('complete', 'complete.json'))}
        paths['remote_receipt'] = self.add(raw / 'retrieval_receipt.json', prefix + '/retrieval_receipt.json', 'readback')
        label = ('ORACLE_' if oracle else 'DEV_') + item['mode']
        fixed = m.get('runner_schema') == 'trajkv.round12.fixed500_static.raw.v1'
        if fixed:
            label = 'FIXED500_R'
        cell = {'cell_id': name, 'label': label, 'dataset': item['domain'],
                'kind': 'pilot' if pilot else 'new_round12',
                'disposition': 'CLOSED_PILOT_EXCLUDED' if pilot else 'CLOSED_COMPLETE',
                'paths': paths, 'producer': frag['producer'],
                'source_graph': frag['source_graph'], 'asset_graph': frag['asset_graph']}
        if pilot:
            cell.update(include_in_quality=False, quality_exclusion_reason='Early real protocol pilot lacks final action runtime coverage; terminal event retained, not a quality source')
        else:
            paths.update(self.data_paths(m))
            paths.update(self.scored(item['scored_dir'], 'results/round12/scored/' + name))
            self.view(label, item['domain'], name)
        require(name not in self.cells, 'fresh producer duplicate: ' + name)
        self.cells[name] = cell

    def bundle(self, raw):
        raw = Path(raw)
        if str(raw) in self.bundles:
            return self.bundles[str(raw)]
        m = read(raw / 'manifest.json')
        prefix = 'evidence/reused_producers/' + sha(raw / 'manifest.json')[:20]
        proofs = [p for p in raw.glob('source_retrieval_receipt*.json')]
        require(len(proofs) == 1, 'original producer proof absent/ambiguous: ' + str(raw))
        proof = read(proofs[0])
        filenames = {'manifest.json', 'records.jsonl', 'complete.json', 'launch_receipt.json', 'source_inventory.json'}
        filenames.update(x['file'] for x in proof['files'])
        files = {}
        for filename in sorted(filenames):
            p = raw / filename
            role = 'code' if p.suffix == '.py' else ('raw' if filename in ('manifest.json','records.jsonl','complete.json') else 'support')
            files[filename] = self.add(p, prefix + '/' + filename, role, p.name != 'records.jsonl')
        proof_public = self.add(proofs[0], prefix + '/' + proofs[0].name, 'readback')
        receipt = raw / 'retrieval_receipt.json'
        if not receipt.exists():
            require(proof['status'] == 'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA', 'no raw readback proof')
            raw_public = proof_public
        else:
            raw_public = self.add(receipt, prefix + '/retrieval_receipt.json', 'readback')
        value = {'files': files, 'source_proof': proof_public, 'raw_receipt': raw_public,
                 'original_evidence_contract_preserved': True, 'new_inference_for_this_stage': False}
        self.bundles[str(raw)] = value
        return value

    def reused(self, c, label_prefix='DEV_'):
        scored = Path(c['scored_dir'])
        s = read(scored / 'summary.json')
        raw = Path(s['integrity']['raw_dir']) if 'integrity' in s else scored.parent.parent / 'raw' / scored.name
        m = read(raw / 'manifest.json')
        domain = s.get('dataset') or m['config']['dataset']
        cell_id = 'REUSED_' + sha(raw / 'manifest.json')[:20]
        label = label_prefix + c['label']
        ids = c.get('ids') or m['expected_ids']
        if cell_id not in self.cells:
            prefix = 'results/round12/reused/' + cell_id
            paths = {k:self.add(raw / n, prefix + '/' + n, 'raw', k != 'records') for k,n in
                     (('manifest','manifest.json'), ('records','records.jsonl'), ('complete','complete.json'))}
            if (raw / 'provenance.jsonl').exists():
                provenance = self.add(raw / 'provenance.jsonl', prefix + '/provenance.jsonl', 'support', False)
                with (raw / 'provenance.jsonl').open(encoding='utf-8-sig') as stream:
                    source_runs = {json.loads(line)['source_run'] for line in stream if line.strip()}
                producer = {'kind': 'legacy_derived_view', 'provenance': provenance,
                            'sources': {run: self.bundle(run) for run in sorted(source_runs)}}
            else:
                producer = {'kind': 'legacy_source_proof', **self.bundle(raw)}
                paths['remote_receipt'] = producer['raw_receipt']
            paths.update(self.data_paths(m, refs=c.get('refs')))
            paths.update(self.scored(scored, 'results/round12/reused_scored/' + cell_id))
            self.cells[cell_id] = {'cell_id':cell_id, 'label': label, 'dataset':domain,
                'kind':'reused_reference', 'disposition':'CLOSED_COMPLETE', 'paths':paths,
                'producer':producer, 'quality_ids':list(ids)}
        else:
            old = self.cells[cell_id]['quality_ids']
            self.cells[cell_id]['quality_ids'] = sorted(set(old) | set(ids))
        self.view(label, domain, cell_id, ids)

    def support(self):
        required = ('code/metrics.py','code/vendor_textvqa_metric.py','code/score.py','code/round12_report_v1.py',
                    'code/round12_report_v2.py','ops/round12_public_core_v1.py','ops/round12_public_legacy_v1.py',
                    'ops/round12_public_contract_v1.py','ops/export_public_round12_v1.py','ops/verify_public_round12_v1.py',
                    'ops/round12_assemble_public_scope_v1.py','ops/publish_public_round12_v1.py',
                    'ops/check_round12_legacy_public_v1.py',
                    'ops/round12_resource_analysis_v1.py',
                    'ops/ROUND12_SCOPE_ASSEMBLY_API_v1.md',
                    'ops/check_round12_publisher_v1.py','ops/ROUND12_PUBLISHER_API_v1.md',
                    'ops/export_public.py','ops/export_public_round9_formal_v6.py')
        for rel in required:
            self.add(ROOT / rel, rel, 'code' if rel.endswith('.py') else 'support')
        for path in sorted((ROOT / 'code').glob('round12*.py')):
            self.add(path, path.relative_to(ROOT).as_posix(), 'code')
        for path in sorted((ROOT / 'third_party/metric_sources').glob('*LICENSE*')):
            self.add(path,path.relative_to(ROOT).as_posix(),'license')
        checks=BASE/'publication_preparation_v1/LEGACY_ACTUAL_CHECKS_v1.json'
        if checks.exists():
            self.add(checks,'evidence/technical/LEGACY_ACTUAL_CHECKS_v1.json','technical')
        for name in ('ROOT_RESOURCE_ANALYSIS_v1.json','ADAPTIVE_SPATIAL_PRECISION_DECISION_v1.json',
                     'LAST32_EXPANSION_DECISION_v1.json','R_FIXED500_SELECTION_EVIDENCE_v1.json',
                     'FINAL_STAGE_CUTOFF_v1.json',
                     'PLAN_ROUND12_EXECUTION_v3_500_ADMISSION.md'):
            if (BASE / name).exists():
                self.add(BASE / name, 'evidence/decisions/' + name, 'support')
        analysis=BASE/'error_pattern_analysis_v1'
        analysis_files=('ERROR_PATTERNS_AND_NEXT_HYPOTHESES_v1.zh.md','ERROR_PATTERNS_v1.json',
                        'ADDITIONAL_PATTERN_VIEWS_v1.json','COMPACT_ANALYSIS_ROWS_v1.jsonl',
                        'analyze_closed_development_v1.py','additional_pattern_views_v1.py')
        if analysis.is_dir():
            support=[]
            for name in analysis_files:
                path=analysis/name
                public='results/round12_analysis/error_pattern_analysis_v1/'+name
                self.add(path,public,'code' if path.suffix=='.py' else 'support')
                support.append({'public_path':public,'source_sha256':sha(path)})
            source=analysis/'COMPACT_ANALYSIS_ROWS_v1.jsonl'
            with source.open(encoding='utf-8-sig') as stream:
                rows=[json.loads(line) for line in stream if line.strip()]
            require(rows,'closed development compact analysis unexpectedly empty')
            csvpath=self.output/'analysis_support/COMPACT_ANALYSIS_ROWS_v1.csv'
            csvpath.parent.mkdir(parents=True,exist_ok=True)
            fields=list(rows[0])
            require(all(set(row)==set(fields) for row in rows),'compact development fields differ')
            with csvpath.open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
                writer.writeheader()
                for row in rows:
                    writer.writerow({key:json.dumps(value,ensure_ascii=False,separators=(',',':'))
                                     if isinstance(value,(dict,list)) else value for key,value in row.items()})
            public='results/round12_analysis/error_pattern_analysis_v1/COMPACT_ANALYSIS_ROWS_v1.csv'
            self.add(csvpath,public,'quality_table')
            manifest={'schema':'trajkv.round12.additional_support.v1',
                'analysis_scope':'posthoc already-closed development only; not new fixed500 evidence',
                'new_GPU_calls':0,'new_task_answers':0,'read_inflight_fixed500_outputs':False,
                'helper_scope':'original local analysis entry sources; public one-command replay is not claimed',
                'original_analysis_files':support,'compact_CSV_source_sha256':sha(source),
                'compact_CSV_sha256':sha(csvpath),'compact_CSV_rows':len(rows),
                'original_analysis_bytes_modified':False}
            path=self.output/'analysis_support/SUPPORT_MANIFEST_v1.json'
            write(path,manifest)
            self.add(path,'results/round12_analysis/SUPPORT_MANIFEST_v1.json','support')
        hull=BASE/'fixed500_average_budget_hull_v1'
        if (hull/'AVERAGE_BUDGET_HULL_v1.json').is_file():
            diagnostic=read(hull/'AVERAGE_BUDGET_HULL_v1.json')
            require(diagnostic.get('status')=='PASS_POSTHOC_COMPLETE500_AVERAGE_BUDGET_DIAGNOSTIC'
                    and diagnostic.get('new_inference')==0 and diagnostic.get('new_GPU_calls')==0,
                    'only completed500 auxiliary hull analysis may enter support')
            prep=BASE/'average_budget_hull_preparation_v1'
            for name in ('round12_average_budget_hull_v1.py','README.zh.md'):
                self.add(prep/name,'results/round12_analysis/average_budget_hull_preparation_v1/'+name,
                         'code' if name.endswith('.py') else 'support')
            for name in ('AVERAGE_BUDGET_HULL_v1.json','SOURCE_COST_SCORE_ROWS_v1.csv','AVERAGE_BUDGET_HULL_v1.zh.md'):
                self.add(hull/name,'results/round12_analysis/fixed500_average_budget_hull_v1/'+name,
                         'quality_table' if name.endswith('.csv') else 'support')
        for _, job in self.producers.values():
            for key in ('launch','job','snapshot_inventory','entrypoint','job_config','parent_complete'):
                if job.get(key):
                    self.entry(job[key])
            for e in job.get('QA0_support_files', []):
                self.entry(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, default=BASE / 'REPORT_ROUND12_FINAL_v1.zh.md')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draft', action='store_true')
    p.add_argument('--check-closed-bindings', action='store_true', help='With --draft, validate only closed producers; cannot seal or publish')
    a = p.parse_args()
    require(a.output.resolve().is_relative_to(BASE.resolve()) and not a.output.exists(), 'new R12 output directory required')
    a.output.mkdir(parents=True)
    b = Assemble(a.output)
    devpath, dev = latest_state(BASE / 'harvest_v2')
    orapath, ora = latest_state(BASE / 'oracle_harvest_v2')
    fixpath, fix = latest_state(BASE / 'fixed500_harvest_v1')
    for obj, oracle in ((dev,False),(ora,True),(fix,False)):
        for item in obj['cells']:
            if obj is fix:
                item = {**item, 'mode':'R'}
            b.new_cell(item, oracle=oracle)
    devplan = read(devpath.parent / 'main_comparison_plan.json')
    for c in devplan['cells']:
        if c['origin'] == 'reused_reference':
            b.reused(c)
    pack = read(BASE / 'fixed500_harvest_v1/reference_pack_v1/REFERENCE_PACK_v1.json')
    for domain, methods in pack['references'].items():
        for label, c in methods.items():
            b.reused({'label':label, 'scored_dir':c['scored_dir']}, 'FIXED500_')
    b.support()
    qa0, attempts = [], []
    for _, job in b.producers.values():
        if job.get('QA0_result'):
            rel = b.entry(job['QA0_result'])
            q = read(job['QA0_result']['source_path'])
            producer={key:job[key]['public_path'] for key in ('launch','job','snapshot_inventory','entrypoint')}
            producer['support_files']=[x['public_path'] for x in job.get('QA0_support_files',[])]
            if job['classification'] == 'QA0_CALIBRATION':
                qa0.append({'kind':'calibration','file':rel,'producer':producer})
            elif job['classification'] == 'QA0_MODEL_DIAGNOSTIC':
                qa0.append({'kind':'model_TF','file':rel,'new_free_QA':0,
                            'actual_model_forwards':q['completed_model_forwards'],'producer':producer})
            else:
                qa0.append({'kind':'packed_cuda','file':rel,'producer':producer})
        if job['classification'] in ('SETUP_OR_PRE_MODEL_FAILURE_RETAIN_ACTUAL_LOG','PREFLIGHT_REJECTED_NOT_LAUNCHED'):
            files = [b.entry(job[k]) for k in ('launch','job','snapshot_inventory','job_config','attempt_log') if job.get(k)]
            if not files:
                path = ROOT / 'configs/jobs' / (job['name'] + '.json')
                files = [b.add(path, 'evidence/attempts/' + job['name'] + '/job.json','attempt')]
            attempts.append({'attempt_id':job['name'],'disposition':'NOT_LAUNCHED' if job['classification'].startswith('PREFLIGHT') else 'PRE_MODEL_FAILED',
                'reason':job['classification'], 'files':files,'successful_terminal_answers':0,
                'actual_model_forwards':0,'recorded_task_rows':0})
    comparisons=[]
    for (label,d),v in b.views.items():
        if label.startswith(('DEV_','ORACLE_')) and label not in ('DEV_FP16','DEV_R7low','DEV_R7high'):
            refs=['DEV_FP16','DEV_R7low','DEV_R7high']
        elif label=='FIXED500_R':
            refs=[l for l,domain in b.views if domain==d and l.startswith('FIXED500_') and l!=label]
        else:
            refs=[]
        for ref in refs:
            if (ref,d) in b.views:
                comparisons.append({'candidate_view':v['view_id'],'reference_view':b.views[(ref,d)]['view_id'],
                                    'bootstrap':2000,'seed':20261124})
    scope={'schema':'trajkv.round12.public_scope.v1','namespace':'external_baselines/baseline_matrix_v3',
        'phase_label':'ROUND12_ACTUAL_FINAL_CUTOFF','fixture_only':False,
        'research_cutoff_complete':not b.blockers,'stage_state':'UNSEALED_DRAFT' if a.draft else 'READY_FOR_ROOT_SEAL',
        'root_seal':None,'whole_external_library_completion_claimed':False,'SourceRun_admission_claimed':False,
        'metric_contract':'matrix_v3_metrics_v1','files':list(b.files.values()),'cells':list(b.cells.values()),
        'views':list(b.views.values()),'comparisons':comparisons,'qa0':qa0,'attempts':attempts,
        'groups':[{'group_id':'development','disposition':'EXECUTED','cell_ids':[n for n,c in b.cells.items() if c['kind']=='new_round12' and c['label'].startswith(('DEV_','ORACLE_'))]},
                  {'group_id':'fixed500_Info_Doc','disposition':'EXECUTED','cell_ids':[n for n,c in b.cells.items() if c['kind']=='new_round12' and c['label']=='FIXED500_R']},
                  *[{'group_id':name,'disposition':'NOT_RUN',
                     'reason':'Frozen R did not outperform R7high quality in both priority domains: Info -1.78 pp; Doc +0.34 pp with paired CI crossing zero. Improve quality before broadening the matrix. This stage did not run full5000.',
                     'cell_ids':[],'primary_mean':None,'actual_terminal_answers':0}
                    for name in ('Round12_R_TextVQA_fixed500','Round12_R_ChartQA_fixed500','Round12_second_model')],
                  {'group_id':'full5000','disposition':'NOT_RUN','reason':'User requested fixed500 comparisons rather than complete5000','cell_ids':[],'primary_mean':None,'actual_terminal_answers':0}],
        'expected_accounting':{'new_producer_rows':2200,'new_successful_terminal_answers':2200,
            'new_failure_status_rows':0,'pilot_producer_rows':2,'pilot_successful_terminal_answers':2},
        'fixed500_is_previously_observed_benchmark_not_independent_confirmation':True}
    write(a.output / 'ASSEMBLY_STATUS.json', {'status':'DRAFT_BLOCKED' if b.blockers else 'ACTUAL_CUTOFF_CLOSED_NOT_SEALED',
        'blockers':sorted(set(b.blockers)), 'cells':len(b.cells),'files':len(b.files), 'source_snapshots':[str(x) for x in (devpath,orapath,fixpath)],
        'GitHub_mutation':False,'GPU_work_started':False})
    if a.draft:
        scope['research_cutoff_complete']=False
        write(a.output/'SCOPE_DRAFT_NOT_SEALABLE.json',scope)
        if a.check_closed_bindings:
            from copy import deepcopy
            check=deepcopy(scope)
            check['research_cutoff_complete']=True
            check['phase_label']='CLOSED_PRODUCER_BINDING_CHECK_NOT_STAGE'
            check['stage_state']='UNSEALED_PARTIAL_TECHNICAL_CHECK'
            check['expected_accounting']={}
            valid={c['cell_id'] for c in check['cells'] if c.get('producer',{}).get('kind')!='sequence_child'
                   or c['producer'].get('parent_complete')}
            check['cells']=[c for c in check['cells'] if c['cell_id'] in valid]
            check['views']=[{**v,'members':[m for m in v['members'] if m['cell_id'] in valid]} for v in check['views']]
            check['views']=[v for v in check['views'] if v['members']]
            view_ids={v['view_id'] for v in check['views']}
            check['comparisons']=[c for c in check['comparisons'] if {c['candidate_view'],c['reference_view']}<=view_ids]
            check['groups']=[]
            result=verify_semantics(check,LocalSource(check,(ROOT,E)))
            result['actual_stage_sealed']=False
            write(a.output/'CLOSED_BINDING_CHECK_NOT_STAGE.json',result)
        print(json.dumps({'status':'DRAFT_NOT_SEALED','output':str(a.output),'blockers':len(set(b.blockers)),'cells':len(b.cells),'files':len(b.files)}))
        return 0
    require(not b.blockers, 'actual cutoff still blocked; inspect ASSEMBLY_STATUS.json')
    require(a.report.is_file(), 'Root final science report not yet written')
    # Recompute the portable report from the exact actual source graph before
    # selecting it as a report_summary. This is never a copied headline table.
    scope['files']=list(b.files.values())
    result=verify_semantics(scope,LocalSource(scope,(ROOT,E)))
    unified={'schema':'trajkv.round12.unified_actual_report.v1','status':'PASS_COMPLETE_DECLARED_CUTOFF',
        'quality':result['quality'],'paired':[{'candidate':next(v['label'] for v in scope['views'] if v['view_id']==p['candidate_view']),
            'reference':next(v['label'] for v in scope['views'] if v['view_id']==p['reference_view']),
            'dataset':next(v['dataset'] for v in scope['views'] if v['view_id']==p['candidate_view']),**p['recomputed']} for p in result['paired']],
        'actual_counts':result['counts'],'fixed500_independent_holdout_claimed':False,
        'new_model_work_for_reused_references':0,'no_cross_domain_macro':True}
    write(a.output/'summary.json',unified)
    qpath=a.output/'quality_complete_bytes.csv'
    with qpath.open('x',encoding='utf-8',newline='') as f:
        fields=['label','dataset','questions','source_clusters','primary_mean','mean_complete_bits','max_complete_bits','mean_complete_visual_bytes','encode_ms','wall_ms']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for q in unified['quality']:
            w.writerow({**{k:q[k] for k in fields[:5]},'mean_complete_bits':(q['actual_effective_visual_bits'] or {}).get('mean'),
                'max_complete_bits':(q['actual_effective_visual_bits'] or {}).get('max'),
                'mean_complete_visual_bytes':(q['stored_visual_KV_bytes'] or {}).get('mean'),
                'encode_ms':(q['encode_ms'] or {}).get('mean'),'wall_ms':(q['wall_ms'] or {}).get('mean')})
    pairpath=a.output/'paired_comparisons.csv'
    with pairpath.open('x',encoding='utf-8',newline='') as f:
        fields=['candidate','reference','dataset','n','primary_delta_pp','ci95_pp','improved','worsened','unchanged','status']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(unified['paired'])
    b.add(a.report,'docs/REPORT_ROUND12_FINAL_v1.zh.md','report')
    b.add(qpath,'results/round12_final/quality_complete_bytes.csv','quality_table')
    b.add(pairpath,'results/round12_final/paired_comparisons.csv','quality_table')
    scope['report_summary']=b.add(a.output/'summary.json','results/round12_final/summary.json','report')
    scope['files']=list(b.files.values())
    finalcheck=verify_semantics(scope,LocalSource(scope,(ROOT,E)))
    write(a.output/'SEMANTIC_CHECK_NOT_SEALED.json',finalcheck)
    write(a.output/'R12_SCOPE_UNSEALED.json',scope)
    print(json.dumps({'status':'READY_FOR_ROOT_SEAL_NOT_PUBLISHED','scope':str(a.output/'R12_SCOPE_UNSEALED.json'),
                      'files':len(b.files),'cells':len(b.cells),'counts':result['counts']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
