"""New hypothesis, no confirmation use: same-byte MAD/sign controlled comparisons."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
probe=json.loads((ROOT/'configs/a2/value_probe_config.json').read_text());probe['variants']=[]
for axis,layout,encoder in [('k','channel','sign'),('k','channel','mad'),('k','channel','lloyd'),
                            ('v','token','mad'),('v','channel','mad'),('v','token','lloyd'),('v','channel','lloyd')]:
 probe['variants'].append({'name':f'{axis}1_{layout}_{encoder}_other_fp16','method':'axisdiag','axis':axis,'axis_bits':1,'layout':layout,'encoder':encoder,'selector':'none'})
(ROOT/'configs/a2/mad_a_probe.json').write_text(json.dumps(probe,indent=2))
dev=json.loads((ROOT/'configs/a2/a2_short_dev4_s1_v1.json').read_text());dev['start']=0;dev['count']=8;dev['variants']=[]
for name,kenc,venc,kb,vb in [('joint_mad','mad','mad',1,1),('k_mad_v_rtn','mad','rtn',1,1),('k_rtn_v_mad','rtn','mad',1,1),
                            ('joint_sign','sign','sign',1,1),('k2_v_mad','rtn','mad',2,1)]:
 dev['variants'].append({'name':name,'method':'rate','bits_k':kb,'bits_v':vb,'k_encoder':kenc,'v_encoder':venc,'cap':8,'store_selector_meta':False})
(ROOT/'configs/a2/mad_a_dev.json').write_text(json.dumps(dev,indent=2))
for name,gpu,cfg in [('mad_a_probe_gpu0_v1',0,'mad_a_probe'),('mad_a_dev_gpu1_v1',1,'mad_a_dev')]:
 j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_variants.py','--config',REMOTE_ROOT+'/configs/a2/'+cfg+'.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/(name+'.json')).write_text(json.dumps(j,indent=2))
print(json.dumps({'single_axis_variants':len(probe['variants']),'joint_variants':len(dev['variants']),'confirmation_started':False}))
