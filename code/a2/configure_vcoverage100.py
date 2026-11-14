"""Low-cost coverage exchange motivated by the new100 V-centroid gain."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 cfg=json.loads((ROOT/'configs/a2/final100_e2e_s0_v1.json').read_text());cfg.update(start=0,count=100);variants=[]
 for label,layers in [('early2',[0,1]),('late2',[26,27])]:
  variants.append({'name':'lloyd_v2_'+label,'method':'rate','bits_k':2,'bits_v':1,'cap':8,'store_selector_meta':False,'v_layout':'channel',
   'layer_bits_v':[2 if i in layers else 1 for i in range(28)],'layer_encoders_v':['rtn' if i in layers else 'lloyd' for i in range(28)]})
 for label,layers in [('late8',list(range(20,28))),('spread8',[0,4,8,12,16,20,24,27])]:
  variants.append({'name':'mad_global_vres_'+label,'method':'rate','bits_k':2,'bits_v':1,'cap':8,'store_selector_meta':False,
    'v_layout':'channel','v_encoder':'mad','v_group_size':'all','v_layers':layers})
 for axis,bits,encoder in [('k',2,'rtn'),('v',1,'lloyd')]:
  variants.append({'name':f'{axis}{bits}_{encoder}_other_fp16','method':'axisdiag','axis':axis,'axis_bits':bits,'encoder':encoder,'layout':'channel','cap':8,'selector':'none','store_selector_meta':False})
 cfg['variants']=variants;name='vcoverage100_gpu1_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2));dep='temperature_global20_gpu1_v1'
 queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
  'tasks':[{'name':'kernel','entrypoint':'check_v_coverage.py','arguments':['--legacy-scripts',cfg['arguments']['legacy_scripts']]},
           {'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
 (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2));print(json.dumps({'images':100,'actual_packed_variants':4,'single_axis_diagnostics':2,'records':600}))
if __name__=='__main__':main()
