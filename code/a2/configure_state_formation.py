"""Matched100 protocol intervention: question states vs first shared logit."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text())
 wanted={'rtn_k1v1','base2_reference','native_k2v1','mad_k1_vmadc','rtn_k1_vmadc_headgain'}
 variants=[v for v in base['variants'] if v['name'] in wanted]
 variants.append({'name':'fp16_reference100','method':'full','cap':8,'selector':'none','store_selector_meta':False})
 for gpu,tag in ((0,'native_history'),(1,'native_first_logit')):
  name=f'state100_{tag}_v1';cfg=copy.deepcopy(base);cfg.update(start=0,count=100,variants=variants,diagnostic_protocol=tag)
  cfg['arguments']['compress_before_question']=gpu==0
  cfg['arguments']['native_question_history']=gpu==0
  (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  dep='geometry20_gpu0_v1' if gpu==0 else 'temperature20_gpu1_v1'
  queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
   'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
 print(json.dumps({'protocols':2,'images':100,'variants':len(variants),'records':200*len(variants),
  'scope':'causal protocol diagnostics; primary quality remains complete question recomputation'}))
if __name__=='__main__':main()
