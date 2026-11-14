"""Correct bit-dependent official clipping, plus explicit bias ablations."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());base.update(start=0,count=100)
 base['variants']=[{'name':'calib2_packed_correct','method':'calib_packed','bits_k':2,'bits_v':2,'clip_k':1.,'clip_v':1.,'tau':-3.,'cap':8,'store_selector_meta':False},
  {'name':'calib_k2v1_packed_correct','method':'calib_packed','bits_k':2,'bits_v':1,'clip_k':1.,'clip_v':.3,'tau':0.,'cap':8,'store_selector_meta':False}]
 (ROOT/'configs/a2/calib_packed100_v1.json').write_text(json.dumps(base,indent=2))
 c=copy.deepcopy(base);c['count']=20;c['variants']=[]
 for center in (False,True):
  for high in (False,True):
   c['variants'].append({'name':f'calib1_center{int(center)}_fp32{int(high)}','method':'calib_packed','bits_k':1,'bits_v':1,'clip_k':.5,'clip_v':.3,'tau':0.,
    'preserve_center_k':center,'score_fp32':high,'cap':8,'store_selector_meta':False})
 (ROOT/'configs/a2/calib_bias20_v1.json').write_text(json.dumps(c,indent=2))
 name='calib_packed_gpu0_v1';dep='fold_gain20_gpu0_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'reference100','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/calib_packed100_v1.json']},
           {'name':'bias20','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/calib_bias20_v1.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
 print(json.dumps({'reference_records':200,'bias_ablation_records':80,'format':'official packed GPU operators, not materialized port'}))
if __name__=='__main__':main()
