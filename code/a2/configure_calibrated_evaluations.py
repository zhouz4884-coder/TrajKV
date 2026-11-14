"""Actual packed validation and a matched coupled-K/V calibration comparison."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def job(name,gpu,dep,tasks):
 q={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json','tasks':tasks}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
def main():
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());base.update(start=0,count=100)
 cfg=copy.deepcopy(base);cfg['variants']=[]
 for tag,gpu in [('cold',0),('warm',1)]:
  for vbits in (1,2):
   cfg['variants'].append({'name':f'calibrated_{tag}_k1v{vbits}','method':'rate','bits_k':1,'bits_v':vbits,'k_encoder':'rtn','k_fold_gain':'head_residual_norm',
    'k_fold_factor_path':REMOTE_ROOT+f'/results/endogenous20_{tag}_gpu{gpu}_v1/calibration/factors.json','v_encoder':'lloyd' if vbits==1 else 'rtn',
    'v_layout':'channel' if vbits==1 else 'token','cap':8,'store_selector_meta':False})
 name='calibrated100_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
 job(name,0,'endogenous20_warm_gpu1_v1',[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}])
 # The two fits share images, targets, parameterization and optimizer steps;
 # the question-state formation protocol is the controlled difference.
 name='coupled_calibration_gpu1_v1';tasks=[]
 for tag,mode in [('cold','compressed_question'),('warm','native_question_history')]:
  fit={'arguments':copy.deepcopy(base['arguments']),'images':20,'steps':200,'learning_rate':.02,'mode':mode,'bits_k':2,'bits_v':1,'train_value':True}
  fit['arguments']['max_patches']=8;path=f'coupled20_{tag}_v1.json';(ROOT/'configs/a2'/path).write_text(json.dumps(fit,indent=2))
  tasks.append({'name':'fit_'+tag,'entrypoint':'calibrate_endogenous.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+path]})
 ev=copy.deepcopy(base);ev['variants']=[]
 for tag in ('cold','warm'):
  factors=REMOTE_ROOT+f'/results/{name}/fit_{tag}/factors.json'
  ev['variants'].append({'name':'coupled_'+tag+'_k2v1','method':'rate','bits_k':2,'bits_v':1,'k_encoder':'rtn','k_fold_gain':'head_residual_norm',
   'k_fold_factor_path':factors,'v_fold_factor_path':factors,'v_encoder':'lloyd','v_layout':'channel','cap':8,'store_selector_meta':False})
 ev['variants'].append({'name':'k2_rms_vlloyd_uncalibrated','method':'rate','bits_k':2,'bits_v':1,'k_encoder':'rtn','k_fold_gain':'head_residual_norm',
  'v_encoder':'lloyd','v_layout':'channel','cap':8,'store_selector_meta':False})
 path='coupled_calibrated100_v1.json';(ROOT/'configs/a2'/path).write_text(json.dumps(ev,indent=2));tasks.append({'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+path]})
 job(name,1,'axis_flip100_gpu1_v1',tasks)
 print(json.dumps({'packed_validation_records':700,'additional_calibration_conditions':2,'joint_parameters':224,'ground_truth_labels':False}))
if __name__=='__main__':main()
