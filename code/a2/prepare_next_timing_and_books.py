"""Finite post-confirmation timing, then isolated next representation calibration."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def write(path,value):
 if path.exists():raise RuntimeError('Do not overwrite staged execution: '+str(path))
 path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def job(name,shard,entry,config):
 result={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/'+entry,'--config',REMOTE_ROOT+'/configs/a2/'+config,'--output',REMOTE_ROOT+'/results/'+name]}
 write(ROOT/'configs/jobs'/f'{name}.json',result)

def main():
 marker=ROOT/'audit/a2/post_confirmation_execution_v1.json'
 if marker.exists():raise RuntimeError('Post-confirmation queue already defined')
 frozen=json.loads((ROOT/'audit/a2/next_confirmation_v1.json').read_text());lookup={v['name']:v for v in frozen['variants']};aliases=frozen['aliases']
 selected=['full8','base2_8','native4_8','k2_lloyd_v1','new_storage_primary','warm_multi_gain','full16','base2_16','native4_16',
  'newread_all','newread_none','newread_freshrisk','newread_cached','newread_prefill_cached','newread_prefill_fresh','newread_periodic']
 variants=[];seen=set()
 for label in selected:
  name=aliases[label]
  if name not in seen:variants.append(copy.deepcopy(lookup[name]));seen.add(name)
 previous=json.loads((ROOT/'configs/a2/frozen_timing_gpu0_v2.json').read_text());previous.update(images=6,repeats=3,variants=variants)
 previous['frozen_contract_sha256']=hashlib.sha256((ROOT/'audit/a2/next_confirmation_v1.json').read_bytes()).hexdigest();previous['declaration']='timing methods fixed before viewing confirm256 or balanced128 outcomes; matched8 outputIDs/question; quality not scored'
 timing_name='next_timing_gpu0_v1';write(ROOT/'configs/a2'/f'{timing_name}.json',previous)
 wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{name}/complete.json','launch':REMOTE_ROOT+f'/jobs/{name}.launch.json'} for name in frozen['jobs']],
  'entrypoint':'run_timing_multiturn.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+timing_name+'.json']}
 wp=timing_name+'_wait.json';write(ROOT/'configs/a2'/wp,wait);job(timing_name,0,'run_after_dependencies.py',wp)
 base=json.loads((ROOT/'configs/a2/trajectory_next32_s0_v1_multi_affine.json').read_text());encoder=next(v['direction_book_path'] for v in frozen['variants'] if v.get('direction_book_path'))
 design=[];book_jobs=[]
 for shard,mode in [(0,'compressed_question'),(1,'native_question_history')]:
  name=f'next_books_s{shard}_v1';tasks=[{'name':'arithmetic','entrypoint':'check_learned_direction.py','arguments':[]}]
  cfg={'arguments':base['arguments'],'images':83,'target_tokens':8,'steps':400,'learning_rate':.003,'regularization':.01,
   'encoder_book_path':encoder,'mode':mode,'train_value':True,'packed_forward':True,'assert_forward_identity':True,
   'headwise_if_budget':True,'book_budget_bits':1.8}
  check=copy.deepcopy(cfg);check.update(images=4,steps=2,target_tokens=3)
  cp=ROOT/'configs/a2'/f'{name}_gradient.json';write(cp,check);tasks.append({'name':'gradient','entrypoint':'calibrate_direction_books.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]})
  for task,joint in [('book_only',False),('joint',True)]:
   config=copy.deepcopy(cfg);config['train_value']=joint;cp=ROOT/'configs/a2'/f'{name}_{task}.json';write(cp,config)
   tasks.append({'name':task,'entrypoint':'calibrate_direction_books.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]})
   design.append({'job':name,'task':task,'mode':mode,'joint_value':joint,'steps':400,'images':32,'cases':83,'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()})
  queue={'wait_complete':REMOTE_ROOT+'/results/'+timing_name+'/complete.json','wait_launch':REMOTE_ROOT+'/jobs/'+timing_name+'.launch.json','tasks':tasks}
  qp=name+'_queue.json';write(ROOT/'configs/a2'/qp,queue);job(name,shard,'run_queue.py',qp);book_jobs.append(name)
 report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'timing_job':timing_name,'timing_images':6,'timing_variants':len(variants),
  'timing_measurements':6*len(variants)*3,'timing_warmups':6*len(variants),'scope':'all four frozen quality jobs must finish, then both-card quiet timing, only then new model calibration',
  'new_book_jobs':book_jobs,'new_book_design':design,'new_book_scope':'future representation study, not retroactively part of current confirm256; no current test score used for this design',
  'calibration':'32already-assigned calibration source docs,83question inputs,noGT,Full self-generated<=8tokens; pretrained model frozen;114688 decoder entries plus224 folded value parameters if joint',
  'forward':'exact packed attention with dense straight-through gradient; numeric and gradient checks must run before fits',
  'budget':'FWHT removes65536-byte matrix; headwise FP16 decoder table229376bytes plus fixed2048-byte encoder; choose global compact decoder on small images if required to stay<=1.8bits',
  'future_validation':'existing open dev64 for selection; reserve another unused source-document set before future confirmation; do not relabel already-running confirm256 as blind for later methods'}
 write(marker,report);print(json.dumps({'timing':timing_name,'measurements':report['timing_measurements'],'warmups':report['timing_warmups'],'book_jobs':book_jobs}))
if __name__=='__main__':main()
