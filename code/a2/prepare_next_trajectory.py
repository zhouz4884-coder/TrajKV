"""Predeclare matched next-phase calibration, before new development outcomes."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(exist_ok=True,parents=True);payload=json.dumps(d,indent=2,ensure_ascii=False).encode('utf-8')
 if p.exists() and p.read_bytes()!=payload:raise RuntimeError('Existing next-phase input differs: '+str(p))
 p.write_bytes(payload)

def main():
 contract=ROOT/'audit/a2/next_trajectory_predevelopment_v1.json'
 if contract.exists():raise RuntimeError('Next-phase contract already exists')
 membership=json.loads((ROOT/'audit/a2/next_docvqa_source_split_v1.json').read_text(encoding='utf-8'))
 sources=json.loads((ROOT/membership['manifests']['cal32']).read_text(encoding='utf-8'));cases=[]
 for row in sources:
  for i,q in enumerate(row['questions'][:3]):
   case=copy.deepcopy(row);case.update(id=row['id']+f'_cal_q{i+1}',session_id=row['id']+f'_cal_q{i+1}',questions=[q],turns=1,
    content=[{'type':'image'},{'type':'text','text':q}],calibration_question_index=i+1)
   cases.append(case)
 path=ROOT/'configs/a2/next_cal32_independent_questions.json';write(path,cases)
 args=json.loads((ROOT/'configs/a2/frozen_timing_gpu0_v1.json').read_text())['arguments']
 args.update(samples=REMOTE_ROOT+'/configs/a2/'+path.name,max_patches=8,turns=1)
 design=[];configs=[];jobs=[]
 for shard,mode in [(0,'compressed_question'),(1,'native_question_history')]:
  name=f'trajectory_next32_s{shard}_v1';tasks=[]
  base={'arguments':args,'images':len(cases),'bits_k':2,'bits_v':1,'train_value':True,'base_rms':False,
   'steps':400,'learning_rate':.01,'regularization':.001,'mode':mode,'target_tokens':8,'train_center':False}
  check=copy.deepcopy(base);check.update(images=4,steps=2,train_center=True,target_tokens=3)
  cfgpath=ROOT/'configs/a2'/f'{name}_gradient_check.json';write(cfgpath,check);configs.append(cfgpath)
  tasks.append({'name':'gradient_check','entrypoint':'calibrate_trajectory_affine.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cfgpath.name]})
  for task,tokens,center in [('first_gain',1,False),('multi_gain',8,False),('multi_affine',8,True)]:
   cfg=copy.deepcopy(base);cfg.update(target_tokens=tokens,train_center=center)
   cfgpath=ROOT/'configs/a2'/f'{name}_{task}.json';write(cfgpath,cfg);configs.append(cfgpath)
   tasks.append({'name':task,'entrypoint':'calibrate_trajectory_affine.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cfgpath.name]})
   design.append({'job':name,'task':task,'mode':mode,'targets':tokens,'parameters':448 if center else 224,'config_sha256':sha(cfgpath)})
  queue={'wait_complete':REMOTE_ROOT+'/results/frozen_timing_gpu0_v1/complete.json','wait_launch':REMOTE_ROOT+'/jobs/frozen_timing_gpu0_v1.launch.json','tasks':tasks}
  qpath=ROOT/'configs/a2'/f'{name}_queue.json';write(qpath,queue);configs.append(qpath)
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+'/configs/a2/'+qpath.name,'--output',REMOTE_ROOT+'/results/'+name]}
  write(ROOT/'configs/jobs'/f'{name}.json',job);jobs.append(name)
 result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'membership_sha256':sha(ROOT/'audit/a2/next_docvqa_source_split_v1.json'),
  'calibration_images':32,'independent_question_cases':len(cases),'labels_used':False,'cases_sha256':sha(path),'design':design,'jobs':jobs,
  'fixed_conditions':'same cases,400 updates, Adam0.01, regularizer0.001, identity K2 initialization, frozen model, all other input and value encoding identical',
  'teacher':'FP16 self-generated answer trajectory up to8 tokens, first1 for first-token arms; no GT answers',
  'center_scope':'add headwise offset in final quantizer scale units to existing minimum; no additional packed decoder tensor',
  'selection':'after evaluating actual packed free generation on new source-disjoint dev64, choose the best <=1.8bit candidate; break ties by conservative bits then name; keep old frozen candidate and strong references; freeze before confirm256',
  'confirmation':'256 source-disjoint documents/999 questions remain unused, selected by fixed ID hash before fitting',
  'source_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'code/a2/calibrate_trajectory_affine.py']},
  'config_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in configs}}
 write(contract,result);print(json.dumps({'calibration_images':32,'cases':len(cases),'fits':len(design),'jobs':jobs,'confirm_questions':999}))
if __name__=='__main__':main()
