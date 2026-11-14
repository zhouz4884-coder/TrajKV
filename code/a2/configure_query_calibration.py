"""Small actual-model calibration, queued without competing GPU workers."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 c=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());c.update(start=0,count=20)
 c['arguments']['max_tokens']=1;c['variants']=[{'name':'query_calibration','method':'querymoments','cap':8,'selector':'none','store_selector_meta':False}]
 name='query_calibration20_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(c,indent=2));dep='vcoverage100_gpu1_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'calibration','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':20,'labels_used':False,'head_dimension':128,'kv_heads':4,'layers':28}))
if __name__=='__main__':main()
