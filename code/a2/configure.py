"""Prospective development configs, independent of new confirmation outcomes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
OLD=REMOTE_ROOT.rsplit('/',1)[0]+'/VisContract_P14_2'
arguments={'model':REMOTE_ROOT.split('/VLAtest/')[0]+'/models/InternVL3-8B-hf',
 'samples':REMOTE_ROOT+'/configs/a2/dev8.json','legacy_scripts':OLD+'/scripts','kivi_source':OLD+'/third_party/KIVI/quant/new_pack.py','max_patches':16}
def write_config(name,variants,count=8):
 c={'schema':'trajkv.a2.variant-config.v1','arguments':arguments,'count':count,'variants':variants}
 p=ROOT/'configs/a2'/(name+'.json');p.write_text(json.dumps(c,indent=2));return p
def job(name,config,gpu=1):
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_variants.py','--config',REMOTE_ROOT+'/'+config.relative_to(ROOT).as_posix(),'--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/(name+'.json')).write_text(json.dumps(j,indent=2))
variants=[{'name':'base2','method':'asym','selector':'none'}, {'name':'all_refined','method':'asym','selector':'all','fraction':1}]
for f in (.25,.5):
 for s in ('random','spread','mass','uncertainty','risk','tie','layer_uniform'):
  variants.append({'name':f'{s}_{f}','method':'asym','selector':s,'fraction':f,'page':256})
p=write_config('dev_selectors',variants);job('a2_select_dev8_gpu1_v1',p)
p=write_config('dev_onebit',[{'name':'base1','method':'base1','selector':'none'},
 {'name':'k_only','method':'asym1_k','selector':'all','fraction':1},
 {'name':'v_only','method':'asym1_v','selector':'all','fraction':1},
 {'name':'kv_refined','method':'asym1','selector':'all','fraction':1},
 {'name':'calibquant1_port','method':'calib1','selector':'none'}]);job('a2_onebit_dev8_gpu1_v1',p)
p=write_config('d0_diag',[{'name':'diagnostic','method':'asym','selector':'none','diagnostic':True}],4);job('a2_diag_dev4_gpu1_v1',p)
# Startup smoke validates every code path on a single development document.
p=write_config('smoke',[
 {'name':'risk','method':'asym','selector':'risk','fraction':.5},
 {'name':'calibquant1_port','method':'calib1','selector':'none'},
 {'name':'compact','method':'base2','selector':'none','compact_history':True,'drop_turns':[2],'drop_before':4}],1)
job('a2_smoke_gpu1_v1',p)
old=json.loads((ROOT/'configs/jobs/a2_grid_dev8_gpu0_v1.json').read_text());old['name']='a2_grid_dev8_gpu0_v2'
old['argv'][-1]=REMOTE_ROOT+'/results/'+old['name'];(ROOT/'configs/jobs'/(old['name']+'.json')).write_text(json.dumps(old,indent=2))
print(json.dumps({'dev_variants':len(variants),'jobs_created':5}))
