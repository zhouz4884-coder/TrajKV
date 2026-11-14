"""Matched112-parameter cold/warm calibration on the same unlabelled20 images."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text())
 for gpu,tag,mode,dep in [(0,'cold','compressed_question','endogenous_smoke_gpu0_v1'),(1,'warm','native_question_history','axis_flip100_gpu1_v1')]:
  name=f'endogenous20_{tag}_gpu{gpu}_v1';cfg={'arguments':base['arguments'],'images':20,'steps':200,'learning_rate':.02,'mode':mode}
  (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
   'tasks':[{'name':'calibration','entrypoint':'calibrate_endogenous.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
  j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
 print(json.dumps({'calibration_images':20,'parameters':112,'steps':200,'conditions':2,'ground_truth_labels':False,'calibration_objective':'FP16 first-answer vocabulary KL; final quality tested in actual packed GPU runtime'}))
if __name__=='__main__':main()
