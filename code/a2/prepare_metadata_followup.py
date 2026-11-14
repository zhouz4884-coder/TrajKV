"""Prespecify low-dimensional calibration and actual packed development."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def write(path,data):
 if path.exists():raise RuntimeError('Metadata follow-up already staged: '+str(path))
 path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def main():
 name='metadata_followup_v1';marker=ROOT/'audit/a2/metadata_followup_design_v1.json'
 if marker.exists():raise RuntimeError('Metadata design already frozen')
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text());lookup={v['name']:v for v in base['variants']}
 template=json.loads((ROOT/'configs/a2/next_books_s0_v2_joint.json').read_text());calibration=[];design=[];variants=[lookup[n] for n in ['full8','native4_8','base2_8','fwht_global','decoder_warm_joint']]
 common=dict(lookup['fwht_global'],method='direction_meta');common.pop('name');common.pop('headwise_if_budget',None);common.pop('book_budget_bits',None)
 variants += [dict(common,name='metadata_identity_g128'),dict(common,name='metadata_identity_all',v_group_size='all')]
 specs=[('g128_v',128,'none',True),('g128_k',128,'centered',False),('g128_kv',128,'centered',True),('g128_wholekv',128,'whole',True),('all_v','all','none',True),('all_kv','all','centered',True)]
 for shard,mode in [(0,'compressed_question'),(1,'native_question_history')]:
  tasks=[{'name':'arithmetic','entrypoint':'check_metadata_direction.py','arguments':[]}]
  check=copy.deepcopy(template);check.update(mode=mode,images=4,steps=2,target_tokens=3,key_gain_mode='whole' if shard==0 else 'centered',train_value=True,value_group_size='all')
  cp=ROOT/'configs/a2'/f'{name}_s{shard}_gradient.json';write(cp,check);tasks.append({'name':'gradient','entrypoint':'calibrate_direction_metadata.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]})
  for label,group,key_mode,train_value in specs:
   cfg=copy.deepcopy(template);cfg.update(mode=mode,key_gain_mode=key_mode,train_value=train_value,value_group_size=group)
   cp=ROOT/'configs/a2'/f'{name}_s{shard}_{label}.json';write(cp,cfg)
   tasks.append({'name':label,'entrypoint':'calibrate_direction_metadata.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]})
   alias=('cold_' if shard==0 else 'warm_')+'meta_'+label
   variants.append(dict(common,name=alias,v_group_size=group,metadata_factor_path=REMOTE_ROOT+f'/results/{name}/calibration_s{shard}/{label}/factors.json'))
   design.append({'variant':alias,'group_size':group,'key_gain_mode':key_mode,'train_value':train_value,'steps':400,'parameters':(112 if key_mode!='none' else 0)+(224 if train_value else 0)})
  qp=ROOT/'configs/a2'/f'{name}_s{shard}_queue.json';write(qp,{'tasks':tasks})
  calibration.append({'name':f'calibration_s{shard}','entrypoint':'run_queue.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+qp.name]})
 development=[]
 for shard in [0,1]:
  config=copy.deepcopy(base);config.update(start=32*shard,count=32,variants=variants)
  cp=ROOT/'configs/a2'/f'{name}_dev_s{shard}.json';write(cp,config)
  development.append({'name':f'development_s{shard}','entrypoint':'run_representation_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]})
 frozen=json.loads((ROOT/'audit/a2/book_confirmation_v1.json').read_text(encoding='utf-8'))
 control={'held_parent_launch':REMOTE_ROOT+'/jobs/direction_floor16_gpu0_v1.launch.json','held_parent_guard':REMOTE_ROOT+'/results/direction_floor16_gpu0_v1/guard.json',
  'dependencies':[{'complete':REMOTE_ROOT+f'/results/{job}/complete.json','launch':REMOTE_ROOT+f'/jobs/{job}.launch.json'} for job in frozen['jobs']],
  'phases':[{'name':'calibration','workers':calibration},{'name':'development','workers':development}]}
 cp=ROOT/'configs/a2'/f'{name}_control.json';write(cp,control)
 job={'name':name,'env':{'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_metadata_followup.py','--config',REMOTE_ROOT+'/configs/a2/'+cp.name,'--output',REMOTE_ROOT+'/results/'+name]}
 write(ROOT/'configs/jobs'/f'{name}.json',job)
 write(marker,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'job':name,'calibration_images':32,'calibration_cases':83,'labels_used':False,'fits':design,
  'development_images':64,'development_variants':len(variants),'development_free_sessions':64*len(variants),'variants':variants,
  'comparison':'fixed global product code; folded scalar K gain and V gain/offset; Vg128 versus whole-image V grouping, no learned decoder table',
  'ordering':'after four existing confirmation jobs, before original V-dose diagnostic then quiet timing; original frozen18 and8 methods unchanged',
  'guards':'hold only idle owned floor waiter, which holds idle timing waiter; two-card child groups terminated on error; finally releases upstream chain',
  'scope':'new open development only; no additional confirmation or standard benchmark primary chosen before complete development'})
 print(json.dumps({'job':name,'fits':len(design),'development_sessions':64*len(variants),'development_variants':len(variants)}))

if __name__=='__main__':main()
