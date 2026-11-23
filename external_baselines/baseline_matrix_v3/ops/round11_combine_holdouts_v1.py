"""Independent pre-packaging combined512 analysis from completed scored roots.

No contract/seal/archive/cold result is required. Default is read-only
readiness; --compute writes derived JSON only after both source cohorts close.
No model/GPU/queue/Git/network or missing-score generation occurs here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
E = Path('${EVIDENCE_ROOT}/results/round11')
FROZEN_COMBINED_HELPER_SHA = 'd9847c1aa1ae325f63022e12c30282c924fe4cf1c5e27523ed684c02142df610'
FROZEN_PUBLIC_LOCK = E/'public_v5_prepare_v1/FROZEN_FOR_ROOT_LOCAL_QA_v1.json'
FROZEN_PUBLIC_LOCK_SHA = 'db21e6d3fb879dc75a0f8ec4e557591e114fe111a0c2fafd65572d79c9bc0617'


def check_frozen_public_epoch():
    """Bind every imported PUB5 dependency to Root's immutable source epoch."""
    body = FROZEN_PUBLIC_LOCK.read_bytes()
    if hashlib.sha256(body).hexdigest() != FROZEN_PUBLIC_LOCK_SHA:
        raise ValueError('frozen PUB5 source-lock epoch changed')
    lock = json.loads(body)
    for name, digest in lock['source_sha256'].items():
        if hashlib.sha256((ROOT/'ops'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen PUB5 source changed: '+name)


check_frozen_public_epoch()
sys.path[:0] = [str(ROOT/'ops'), str(ROOT/'code')]
import round11_public_contract_v5 as public
import round11_public_expansion_v5 as expansion
from round11_public_metric_contracts_v5 import verify_metric_fields, CANONICAL_METRIC_GROUP

SPECS = {
    'original128': {'plan_schema': 'trajkv.round11.confirmation_harvest_plan.v2', 'summary_schema': 'trajkv.round11.confirmation_harvest_summary.v2', 'status': 'PASS_FRESH_R128_ALL_CELLS',
        'QA': 640, 'refs': 384, 'counts': dict(textvqa=48, infovqa=48, docvqa=16, chartqa=16), 'protocol': 'fresh_R128_confirmation', 'adapter': 'round11_confirmation_harvest_v2', 'proof': 'source_retrieval_receipt_round11_confirmation_v2.json'},
    'additional384': {'plan_schema': 'trajkv.round11.expand384_harvest_plan.v1', 'summary_schema': 'trajkv.round11.expand384_harvest_summary.v1', 'status': 'PASS_FRESH_EXPAND384_ALL_CELLS',
        'QA': 1920, 'refs': 1152, 'counts': dict(textvqa=144, infovqa=144, docvqa=48, chartqa=48), 'protocol': 'fresh_additional384_expansion', 'adapter': 'round11_expand_harvest_v1', 'proof': 'source_retrieval_receipt_round11_expand384_v1.json'},
}


class LocalSourceResolver:
    """Only explicitly source-verified local files can reach frozen combine."""
    def __init__(self, files): self.files = {Path(path).resolve(): digest for path, digest in files.items()}
    def path(self, value):
        p = Path(value).resolve(); public.require(p in self.files and public.sha(p) == self.files[p], 'combined analysis opened unbound or changed local file'); return p
    def doc(self, value): return public.load(self.path(value))


def readiness(original_plan, original_root, original_summary, additional_plan, additional_root, additional_summary):
    check_frozen_public_epoch()
    public.require(public.sha(ROOT/'ops/round11_public_expansion_v5.py') == FROZEN_COMBINED_HELPER_SHA, 'frozen combined analysis helper epoch changed')
    source_inputs = {'original128': (Path(original_plan), Path(original_root), Path(original_summary)), 'additional384': (Path(additional_plan), Path(additional_root), Path(additional_summary))}
    missing = []; loaded = {}
    for label, (planpath, root, summarypath) in source_inputs.items():
        spec = SPECS[label]
        for kind, path in (('plan', planpath), ('summary', summarypath)):
            if not path.is_file(): missing.append(label+': missing '+kind)
        if not planpath.is_file() or not summarypath.is_file(): continue
        p, s = public.load(planpath), public.load(summarypath)
        public.require(p.get('schema') == spec['plan_schema'], label+': wrong source plan schema')
        public.require(s.get('schema') == spec['summary_schema'] and s.get('plan_sha256') == public.sha(planpath), label+': wrong summary/actual-plan binding')
        if s.get('status') != spec['status'] or s.get('actual_closed_fresh_QA') != spec['QA']:
            missing.append(label+': actual complete '+str(spec['QA'])+' source QA not closed'); continue
        public.require(s['expected_new_QA'] == spec['QA'] and s['expected_fresh_reference_QA'] == spec['refs'] and s['reused_reference_views'] == 0 and len(s['points']) == len(p['entries']) == 20, label+': full fresh source accounting differs')
        lookup = {(v['dataset'], v['mode']): v for v in s['points']}
        public.require(len(lookup) == 20 and set(lookup) == {(d, m) for d in spec['counts'] for m in ('FP16', 'R7low', 'R7high', 'JointV', 'JointSpatialWO')}, label+': exact five-method matrix differs')
        for entry in p['entries']:
            name = entry['item']['cell']['name']; raw, scored = root/'raw'/name, root/'scored'/name
            files = (raw/'manifest.json', raw/'records.jsonl', raw/'complete.json', raw/spec['proof'], scored/'summary.json', scored/'scored_records.jsonl')
            for f in files:
                if not f.is_file(): missing.append(label+': incomplete cell '+name+'/'+f.name)
            point = lookup[entry['dataset'], entry['mode']]
            if point.get('primary_mean') is None or point.get('fresh_QA') != spec['counts'][entry['dataset']]: missing.append(label+': noncomplete primary cell '+name)
        loaded[label] = (p, s, root, planpath, summarypath)
    if missing: return {'status': 'PENDING_COMPLETE_ORIGINAL128_AND_ADDITIONAL384_SOURCE_COHORTS', 'missing': missing, 'output_files_written': 0, 'new_inference_events': 0}, None
    return {'status': 'READY_FOR_INDEPENDENT_PREPACKAGING_COMBINED512_ANALYSIS', 'missing': [], 'output_files_written': 0, 'new_inference_events': 0}, loaded


def compute(loaded):
    check_frozen_public_epoch()
    from importlib import import_module
    import metrics
    import score
    import verify_public_round10_v3 as physical
    bindings = {}; phase = {'groups': []}; all_rows = {}; snapshots = {}
    for label, (plan, summary, root, planpath, summarypath) in loaded.items():
        spec = SPECS[label]; adapter = import_module(spec['adapter']); adapter.check_pins(); checked, _ = adapter.load_plan(planpath)
        public.require(checked == plan, label+': adapter source plan changed')
        public.require(plan['selection_manifest_sha256'] == expansion.SELECTION_SHA, label+': original selected methods changed')
        for relative, digest in summary['pins'].items(): public.require(public.sha(ROOT/relative) == digest, 'frozen scoring/source pin differs: '+relative)
        snapshots[planpath] = public.sha(planpath); snapshots[summarypath] = public.sha(summarypath)
        points = {(p['dataset'], p['mode']): p for p in summary['points']}; cells = []; all_rows[label] = {}
        for entry in plan['entries']:
            ds, mode = entry['dataset'], entry['mode']; count = spec['counts'][ds]; name = entry['item']['cell']['name']
            raw, scored = root/'raw'/name, root/'scored'/name
            m, done, cfg = adapter.check_source(raw, entry['item']); point = public.load(scored/'summary.json')
            public.require(public.canonical(point) == public.canonical(points[ds, mode]) and point['fresh_QA'] == point['expected_QA'] == count and point['reused_reference_views'] == 0, 'combined source scored point is not the completed frozen group point')
            public.require(public.sha(scored/'scored_records.jsonl') == point['scored_records_sha256'], 'combined scored body differs')
            raw_lines = (raw/'records.jsonl').read_bytes().splitlines(keepends=True); rr = [json.loads(line) for line in raw_lines]
            sr = [json.loads(line) for line in (scored/'scored_records.jsonl').read_text(encoding='utf-8-sig').splitlines() if line]
            refs = {r['id']: r for r in map(json.loads, Path(plan['domains'][ds]['refs']).read_text(encoding='utf-8-sig').splitlines())}
            public.require(len(rr) == len(sr) == count and done['counts'] == {'PASS': count}, 'combined needs full genuine source generation')
            for number, (r, s, line) in enumerate(zip(rr, sr, raw_lines), 1):
                public.require(r['id'] == s['id'] and s['raw_line_number'] == number and s['raw_line_sha256'] == hashlib.sha256(line).hexdigest() and s['score_status'] == 'SCORED_PASS', 'combined score is not exact original raw line')
                verify_metric_fields(metrics, ds, r['raw_prediction'], refs[r['id']], s, CANONICAL_METRIC_GROUP)
            public.require(public.canonical(score.summarize_group(sr, 2000, 20261123)) == public.canonical(point['groups']['overall']), 'combined source aggregate/CI differs from frozen original or additional score')
            all_rows[label][ds, mode] = rr
            for f in (raw/'manifest.json', raw/'records.jsonl', raw/'complete.json', scored/'summary.json', scored/'scored_records.jsonl'):
                snapshots[f] = public.sha(f)
            cells.append({'dataset': ds, 'mode': mode, 'disposition': 'CLOSED_COMPLETE', 'raw': str(raw), 'scored': str(scored)})
        phase['groups'].append({'group_id': label, 'protocol': spec['protocol'], 'execution_disposition': 'EXECUTED', 'selected_modes': list(expansion.CANDIDATES), 'cells': cells})
        bindings[label] = {'source_plan': str(planpath), 'source_plan_sha256': public.sha(planpath), 'source_summary': str(summarypath), 'source_summary_sha256': public.sha(summarypath), 'harvest_root': str(root), 'actual_fresh_QA': spec['QA']}
    for ds in SPECS['original128']['counts']:
        a = all_rows['original128'][ds, 'FP16']; b = all_rows['additional384'][ds, 'FP16']
        for key in ('id', 'source_cluster_id', 'original_image_sha256'):
            public.require(not ({r[key] for r in a} & {r[key] for r in b}), 'combined source intersection: '+key)
        public.require(not ({r['source_id'] for r in a} & {r['source_id'] for r in b}), 'combined source-id intersection')
    result = expansion.combined_512_views(LocalSourceResolver(snapshots), phase, score, physical)
    public.require(result['status'] == 'PASS_DERIVED_ORIGINAL128_PLUS_ADDITIONAL384_COMBINED512' and result['new_inference_events'] == 0 and len(result['points']) == 20 and len(result['pairs']) == 24, 'frozen combined function did not produce exact complete report')
    public.require(all(public.sha(path) == digest for path, digest in snapshots.items()), 'immutable source changed during independent analysis')
    check_frozen_public_epoch()
    return {'schema': 'trajkv.round11.independent_prepackaging_combined512.v1', 'status': result['status'], 'new_inference_events': 0,
        'original128_additional384_combined512_separate_report': result, 'source_bindings': bindings,
        'producer': {'source_sha256': public.sha(__file__), 'frozen_combined_helper_sha256': FROZEN_COMBINED_HELPER_SHA,
            'frozen_public_source_lock_sha256': FROZEN_PUBLIC_LOCK_SHA,
            'requires_final_contract_or_COLD_result': False, 'new_model_GPU_or_inference_events': 0}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--original-plan', type=Path, default=E/'bootstrap_v1/CONFIRM_HARVEST_PLAN_v3.json'); p.add_argument('--original-root', type=Path, default=E/'confirmation_harvest_v3'); p.add_argument('--original-summary', type=Path, default=E/'confirmation_summary_final_v3/summary.json')
    p.add_argument('--additional-plan', type=Path, default=E/'bootstrap_v1/EXPAND384_HARVEST_PLAN_v1.json'); p.add_argument('--additional-root', type=Path, default=E/'expand_harvest_v1'); p.add_argument('--additional-summary', type=Path, default=E/'expand_summary_final_v1/summary.json'); p.add_argument('--output', type=Path, required=True); p.add_argument('--compute', action='store_true'); a = p.parse_args()
    ready, loaded = readiness(a.original_plan, a.original_root, a.original_summary, a.additional_plan, a.additional_root, a.additional_summary)
    if loaded is None or not a.compute: print(json.dumps(ready)); return 0 if loaded is not None else 3
    target = a.output.resolve(); public.require(target.is_relative_to(E.resolve()) and not target.exists(), 'new unique E combined report required')
    result = compute(loaded)
    public.require(not target.exists(), 'combined target appeared during analysis')
    public.write_new(target, result)
    print(json.dumps({'status': result['status'], 'report': str(target), 'sha256': public.sha(target), 'new_inference_events': 0})); return 0


if __name__ == '__main__': raise SystemExit(main())
