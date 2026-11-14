"""Freeze static head orders using calibration-only finite interventions."""
import hashlib,json,random,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from remote import connect,upload,REMOTE_ROOT as R,PYTHON as P


def main():
    source=ROOT/'results/round5_v1/refinement_gain/local_metrics.json'
    rows=json.loads(source.read_text(encoding='utf8'))
    ranks={};scores={}
    for axis in ('k','v'):
        groups=defaultdict(list)
        for row in rows:
            groups[(row['session_id'],row['layer'],row['phase'])].append(row[axis+'_sign_refinement_gain_by_kv_head'])
        per_doc=defaultdict(dict)
        for (sid,layer,phase),values in groups.items():
            per_doc[(sid,layer)][phase]=[sum(v[h] for v in values)/len(values) for h in range(4)]
        by_unit=defaultdict(list)
        for (sid,layer),phases in per_doc.items():
            for h in range(4):
                by_unit[(layer,h)].append(sum(p[h] for p in phases.values())/len(phases))
        scores[axis]={u:sum(v)/len(v) for u,v in by_unit.items()}
        ranks[axis]=sorted(scores[axis],key=lambda u:(-scores[axis][u],u))
    spread=[]
    for i in range(128):
        index=int(f'{i:07b}'[::-1],2)
        if index<112:spread.append((index//4,index%4))
    random_order=[(i//4,i%4) for i in range(112)]
    random.Random(1212).shuffle(random_order)
    folder=ROOT/'configs/a2/round5_v1'
    cfg=json.loads((folder/'v_dev16.json').read_text(encoding='utf8'))
    previous=next(v for v in cfg['variants'] if v['name']=='cold_meta_all_kv')
    variants=[]
    for name,axis,order,limit in [('sparse_none','v',ranks['v'],0),
            ('v_impact_max','v',ranks['v'],None),('v_spread_max','v',spread,None),
            ('v_random_max','v',random_order,None),('k_impact_max','k',ranks['k'],None),
            ('v_impact_10','v',ranks['v'],10),('v_impact_20','v',ranks['v'],20)]:
        v=dict(previous,name=name,method='sparse_v_residual',selected_units=order,
               residual_axis=axis,budget_bits=1.8,
               selection_calibration_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if limit is not None:v['residual_units_limit']=limit
        variants.append(v)
    cfg['variants']=variants
    outputs={'sparse_dev16':cfg,
        'sparse_rankings':{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'same teacher q/text; quantized K/V baseline, one finite sign-residual intervention; equal doc and phase weights',
            'ranks':{axis:[{'unit':list(u),'estimated_local_gain':scores[axis][u]} for u in ranks[axis]] for axis in ranks}}}
    tasks=[dict(name='arithmetic',entrypoint='sparse_v_residual_check.py',arguments=[]),
           dict(name='development16',entrypoint='round5_runner.py',arguments=['--config',R+'/configs/a2/round5_v1/sparse_dev16.json'])]
    outputs['sparse_queue']=dict(tasks=tasks)
    outputs['job_sparse_queue']=dict(name='round5_sparse_queue_v1',argv=[P,R+'/code/a2/run_queue.py','--config',R+'/configs/a2/round5_v1/sparse_queue.json','--output',R+'/results/round5_v1/sparse_queue'],env={'CUDA_VISIBLE_DEVICES':'1'})
    for name,data in outputs.items():
        path=folder/(name+'.json')
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(data,indent=2),encoding='utf8')
    c=connect()
    try:
        paths=[folder/(name+'.json') for name in outputs]+[ROOT/'code/a2'/n for n in ('sparse_v_residual_bank.py','sparse_v_residual_attention.py','sparse_v_residual_check.py','round5_runner.py')]
        for path in paths:upload(c,path,R+'/'+path.relative_to(ROOT).as_posix())
    finally:c.close()
    print(json.dumps({'variants':len(variants),'positive_mean_units':{a:sum(x>0 for x in scores[a].values()) for a in ranks}}))


if __name__=='__main__':main()
