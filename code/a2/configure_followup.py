"""Fixed-cost refinement and static equal-storage grids; no confirmation access."""
from pathlib import Path
import copy,json,math,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
c0=json.loads((ROOT/'configs/a2/dev_selectors.json').read_text())
def config(name,variants):
 c=copy.deepcopy(c0);c['variants']=variants;p=ROOT/'configs/a2'/(name+'.json');p.write_text(json.dumps(c,indent=2));return REMOTE_ROOT+'/'+p.relative_to(ROOT).as_posix()
bounded=[]
for s in ('random','spread','mass','risk','layer_uniform'):
 bounded.append({'name':s+'_bounded','method':'asym','selector':s,'fraction':.25,'global_read_equivalents':32,'page':256})
cal=[{'name':'axis_calibration','method':'asym1','selector':'all','fraction':1,'store_selector_meta':False,'local_calibration':True}]
grid=[]
def layers(n,offset=0):return sorted({(int((j+.5)*28/n)+offset)%28 for j in range(n)}) if n else []
for slots in (12,24,36):
 seen=set()
 for share in (0,.25,.5,.75,1):
  nk=min(28,max(math.ceil(max(0,slots-28)/2),int(round(slots*share/2))));nv=slots-2*nk
  if (nk,nv) in seen:continue
  assert 0<=nv<=28 and 2*nk+nv==slots;seen.add((nk,nv))
  grid.append({'name':f'alloc_s{slots}_k{nk}_v{nv}','method':'alloc1','selector':'all','fraction':1,
    'store_selector_meta':False,'k_layers':layers(nk),'v_layers':layers(nv,14),'slots':slots,
    'expected_effective_visual_bits':1.25+slots*(.5625/28)})
paths=[('bounded','run_variants.py',['--config',config('dev_bounded',bounded)]),
       ('axis_calib','run_variants.py',['--config',config('dev_axis_calib',cal)]),
       ('grid','run_variants.py',['--config',config('dev_alloc_grid',grid)])]
queue={'tasks':[{'name':n,'entrypoint':e,'arguments':v} for n,e,v in paths]}
(ROOT/'configs/a2/refine_queue.json').write_text(json.dumps(queue,indent=2))
j={'name':'a2_refine_dev_gpu1_v1','env':{'CUDA_VISIBLE_DEVICES':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
 'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+'/configs/a2/refine_queue.json','--output',REMOTE_ROOT+'/results/a2_refine_dev_gpu1_v1']}
(ROOT/'configs/jobs'/(j['name']+'.json')).write_text(json.dumps(j,indent=2))
print(json.dumps({'bounded_variants':len(bounded),'equal_byte_grid':len(grid),'effective_bits':[1.25+s*.5625/28 for s in (12,24,36)]}))
