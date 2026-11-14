"""Global attention-temperature controls paired with the visual-only sweep."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 cfg=json.loads((ROOT/'configs/a2/temperature20_v1.json').read_text());variants=[]
 for enc in ('rtn','mad'):
  for beta in (.125,.26,.5,2.):
   variants.append({'name':f'{enc}_global_beta{beta:g}','method':'rate','bits_k':1,'bits_v':1,'k_encoder':enc,'v_encoder':'mad','v_layout':'channel',
    'cap':8,'store_selector_meta':False,'visual_logit_gain':beta,'text_logit_gain':beta})
 cfg['variants']=variants;name='temperature_global20_gpu1_v1'
 (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
 dep='state100_native_first_logit_v1'
 queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));print(json.dumps({'global_control_records':20*len(variants)}))
if __name__=='__main__':main()
