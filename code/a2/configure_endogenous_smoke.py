"""One-image actual-GPU gradient feasibility after current free-run work."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());cfg={'arguments':base['arguments'],'images':1,'steps':3,'learning_rate':.02}
 name='endogenous_smoke_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='fold100_gpu0_v1'
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'feasibility','entrypoint':'calibrate_endogenous.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
if __name__=='__main__':main()
