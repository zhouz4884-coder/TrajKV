"""Freeze the initial 16-document round-five development comparisons."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from remote import connect,upload,REMOTE_ROOT as R,PYTHON as P


def main():
    f=ROOT/'configs/a2/round5_v1'
    old=json.loads((ROOT/'configs/a2/metadata_followup_v1_dev_s0.json').read_text(encoding='utf8'))
    rows=json.loads((ROOT/'configs/a2/docvqa_next_dev64.json').read_text(encoding='utf8'))
    def save(name,value):
        path=f/(name+'.json')
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(value,indent=2),encoding='utf8')
    save('dev16',rows[:16])
    previous=next(v for v in old['variants'] if v['name']=='cold_meta_all_kv')
    common=dict(previous,method='query_spectrum_port',spectrum_artifact_path=R+'/results/round5_v1/spectrum_analysis/transforms.pt',basis_dtype='float32',gain_rule='original_space_norm',collect_score_metrics=True,score_metric_decode_steps=2)
    control=[dict(old['variants'][0])]
    for transform in ['fwht','k_pca','query_aware_mixed']:
        control.append(dict(common,name='roundtrip_'+transform,transform=transform,key_encoding='roundtrip',rank=128,bits_v=16,apply_key_metadata=False,collect_score_metrics=False))
    args=dict(old['arguments'],samples=R+'/configs/a2/round5_v1/dev16.json')
    save('spectrum_roundtrip',dict(old,arguments=args,variants=control,count=2,start=0))
    variants=[old['variants'][i] for i in [0,1,2]]+[previous]
    for transform in ['fwht','random','k_pca','query_aware_mixed']:
        variants.append(dict(common,name='pq_'+transform,transform=transform,key_encoding='pq',rank=128))
    for rank in [4,8,16,32,64]:
        variants.append(dict(common,name='project_query_r'+str(rank),transform='query_aware_mixed',key_encoding='projection',rank=rank))
    for transform in ['k_pca','random']:
        variants.append(dict(common,name='project_'+transform+'_r16',transform=transform,key_encoding='projection',rank=16))
    evalc=dict(old,arguments=args,variants=variants,count=16,start=0)
    save('spectrum_dev16',evalc)
    q=dict(wait_complete=R+'/results/round5_v1/spectrum_analysis/complete.json',wait_launch=R+'/jobs/round5_spectrum_analysis_v1.launch.json',tasks=[dict(name='roundtrip',entrypoint='round5_runner.py',arguments=['--config',R+'/configs/a2/round5_v1/spectrum_roundtrip.json']),dict(name='development16',entrypoint='round5_runner.py',arguments=['--config',R+'/configs/a2/round5_v1/spectrum_dev16.json'])])
    save('spectrum_queue',q)
    save('job_spectrum_queue',dict(name='round5_spectrum_queue_v1',argv=[P,R+'/code/a2/run_queue.py','--config',R+'/configs/a2/round5_v1/spectrum_queue.json','--output',R+'/results/round5_v1/spectrum_queue'],env={'CUDA_VISIBLE_DEVICES':'1'}))
    variants=[old['variants'][i] for i in [0,1,2]]+[previous]
    for g in [1,8]:
        for b in [0,1]:
            key='g%d_b%d'%(g,b)
            variants.append(dict(previous,name=key,method='direction_functional_v',functional_v_factor_path=R+'/results/round5_v1/v_queue/'+key+'/factors.json'))
    save('v_dev16',dict(evalc,variants=variants))
    q=dict(wait_complete=R+'/results/round5_v1/v_queue/complete.json',wait_launch=R+'/jobs/round5_v_queue_v1.launch.json',tasks=[dict(name='development16',entrypoint='round5_runner.py',arguments=['--config',R+'/configs/a2/round5_v1/v_dev16.json'])])
    save('v_eval_queue',q)
    save('job_v_eval',dict(name='round5_v_eval_v1',argv=[P,R+'/code/a2/run_queue.py','--config',R+'/configs/a2/round5_v1/v_eval_queue.json','--output',R+'/results/round5_v1/v_eval'],env={'CUDA_VISIBLE_DEVICES':'0'}))
    c=connect()
    try:
        paths=list(f.glob('*.json'))+[ROOT/'code/a2/query_spectrum_math.py',ROOT/'code/a2/query_spectrum_port.py',ROOT/'code/analyze_query_spectrum_round5.py']
        for path in paths:upload(c,path,R+'/'+path.relative_to(ROOT).as_posix())
    finally:c.close()
    print(json.dumps({'spectrum_variants':len(evalc['variants']),'v_variants':len(variants),'dev_images':16}))


if __name__=='__main__':main()
