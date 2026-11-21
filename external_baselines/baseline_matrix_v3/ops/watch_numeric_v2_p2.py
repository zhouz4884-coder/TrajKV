"""Restricted24 new-port CPU harvester wrapper; original collectors untouched."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MIRROR=Path('${EVIDENCE_ROOT}')
sys.dont_write_bytecode=True


def validate(ledger_dir):
    from prediction_cache import read,sha,selected_inputs
    paths=list(Path(ledger_dir).glob('ledger_v*.json'));assert len(paths)==1,'This fixed24 wrapper expects its one immutable reviewed ledger'
    ledger=read(paths[0]);assert ledger['run_id']=='calib_numeric_v2' and ledger['target_cells']==24
    assert ledger['source_scope']=='configs/matrix/P2_complete114_Calib_FP32Affine_Text32_v2.json'
    assert sha(ROOT/ledger['source_scope'])==ledger['source_scope_sha256']=='fb816d1680734c32ce1474eb8dcc89a860f703ca772783d371311e785f801d5a'
    rows=ledger['rows'];assert len(rows)==24 and ledger['actual_cached_QA']==0 and ledger['planned_fresh_QA']==12000
    expected=set()
    for item in read(ROOT/ledger['source_scope'])['rows']:
        path=item.get('config') or item.get('full_target_config') or item.get('target_config')
        if read(ROOT/path)['method']['family']=='calibquant':expected.add(path)
    assert {r['full_target_config'] for r in rows}==expected and len({r['remote_new_output'] for r in rows})==24
    for row in rows:
        path=ROOT/row['full_target_config'];cfg=read(path)
        assert sha(path)==row['full_target_config_sha256'] and cfg['count']==500
        assert cfg['method']['adapter_module']=='methods.calibquant_fp32_text_adapter' and cfg['method']['adapter_class']=='CalibQuantFP32AffineTextAdapter'
        assert row['cached_runs']==[] and row['cached_PASS_QA']==0 and row['fresh_QA']==500
        relative=row['remote_new_output'].split('/results/',1)[1]
        assert relative.startswith('p2_calib_numeric_v2/') and '..' not in Path(relative).parts
        assert Path(row['reference_run']).resolve()==(MIRROR/'results'/relative).resolve()
        plan=read(row['plan']);assert sha(row['plan'])==row['plan_sha256']
        assert plan['schema']=='baseline_matrix.fresh_prediction_production_plan.v1' and plan['actual_reference_run'] is None and plan['actual_effective_contract_sha256'] is None
        assert plan['cached_PASS_QA']==0 and plan['missing_QA']==500
        assert plan['expected_ids']==[r['id'] for r in selected_inputs(cfg,ROOT/f"configs/datasets/v3/{row['dataset']}_p2_inputs.jsonl")]
    return ledger


def main():
    p=argparse.ArgumentParser();p.add_argument('--ledger-dir',type=Path,default=MIRROR/'results/extension_execution/calib_numeric_v2')
    p.add_argument('--validate-only',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=30)
    a=p.parse_args();sys.path.insert(0,str(ROOT/'code'));validate(a.ledger_dir)
    if a.validate_only:print(json.dumps({'status':'PASS_FRESH_ONLY24_LEDGER_METADATA_NO_NETWORK','GPU':False}));return
    # The generic unchanged collector creates a SourceRun only after actual raw
    # closure, then uses that same fresh500 raw as its own reference/context.
    from watch_extension_quality import main as generic
    sys.argv=[sys.argv[0],'--ledger-dir',str(a.ledger_dir),'--run-id','calib_numeric_v2',
        '--output',str(MIRROR/'results/extension_execution/calib_numeric_v2/harvest'),'--interval',str(a.interval),
        '--watch' if a.watch else '--once']
    generic()


if __name__=='__main__':main()
