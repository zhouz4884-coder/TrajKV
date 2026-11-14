"""Promising K1 direction encoder with V2; query-metric and amplitude controls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=20);vs=[]
 for power in (0.,.5,1.):
  for gain in ('norm','projection'):
   v={'name':f'rotation_v2token_q{power:g}_{gain}','method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,
    'center':True,'rotate':True,'gain':gain,'bits_v':2,'v_layout':'token'}
   if power:v.update(query_scale_path=REMOTE_ROOT+'/configs/a2/query_scales20.json',query_scale_power=power)
   vs.append(v)
 for gain in ('norm','projection'):
  vs.append({'name':'rotation_v2channel_'+gain,'method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,
    'center':True,'rotate':True,'gain':gain,'bits_v':2,'v_layout':'channel'})
 vs.append({'name':'rotation_projection_vfp16','method':'rotk_port','cap':8,'selector':'none','store_selector_meta':False,'center':True,'rotate':True,'gain':'projection'})
 cfg['variants']=vs;name='rotation_v2_probe20_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='coupled_calibration_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':20,'quality_prototypes':len(vs),'actual_packed_efficiency_claims':False}))
if __name__=='__main__':main()
