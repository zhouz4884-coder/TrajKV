"""Shared second-cycle free generation runner, config-frozen before launch."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
import session
import session_boundary
from masked_bank import A2Bank
from calib_bank import CalibBank
from allocation_bank import AllocationBank
from rate_bank import RateBank
from calib_post import CalibPostBank
from hybrid_bank import HybridBank
from single_axis_bank import SingleAxisBank
from hybrid_packed_bank import HybridPackedBank
from factory import bank_class
from transformers import AutoProcessor,AutoModelForImageTextToText
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
 a.output.mkdir(parents=True,exist_ok=False)
 for k,v in c['arguments'].items():setattr(a,k,Path(v) if k in ('model','legacy_scripts','kivi_source','samples') else v)
 for k,v in {'turns':6,'max_tokens':32,'max_patches':16,'margin':0.,'max_replays':0,'save_states':False,'read_budget_bytes':None,'gain_table_data':None}.items():
  if not hasattr(a,k):setattr(a,k,v)
 session.install_legacy(a.legacy_scripts);torch.set_num_threads(4);torch.manual_seed(1212);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text());rows=rows[c.get('start',0):c.get('start',0)+c.get('count',len(rows))]
 manifest={'config':c,'config_sha256':sha(a.config),'samples_sha256':sha(a.samples),'selected_ids':[r['id'] for r in rows],
  'answers_visible_to_runner':False,'sources':{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
 manifest['external_codec_inputs']={v[k]:sha(v[k]) for v in c['variants'] for k in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path','affine_factor_path','decoder_book_path') if v.get(k)}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 n=0
 with torch.inference_mode(),(a.output/'records.jsonl').open('w') as log:
  for row in rows:
   for v in c['variants']:
    cls=bank_class(v['method']);cls.settings=v
    executor=session_boundary if getattr(a,'compress_before_question',False) else session
    if v['method'].startswith('anchor_'):
     import session_anchor_v2
     executor=session_anchor_v2
    executor.ProgressiveBank=cls
    a.max_patches=v.get('cap',c['arguments'].get('max_patches',16))
    t=time.perf_counter();r,_=executor.run_session(model,proc,row,a,v['method'],'base',keep_reference=v.get('diagnostic',False) or v.get('local_calibration',False))
    r['variant']=v['name'];r['variant_config']=v;r['max_patches']=a.max_patches
    if c.get('diagnostic_protocol'):
     r['diagnostic_protocol']=c['diagnostic_protocol'];r['primary_quality_eligible']=False
    if v['method']=='learned_direction':
     r['kernel_variant']='matrix-freeFWHT product code and shared decoder tables';r['kernel_source_sha256']=sha(Path(__file__).with_name('fused_learned_direction.py'))
    elif v['method'] in ('direction','anchor_direction'):
     r['kernel_variant']='packed direction keys, FP32 visual mean restoration';r['kernel_source_sha256']=sha(Path(__file__).with_name('fused_direction.py'))
    elif v['method']=='calib_packed':
     r['kernel_variant']='official CalibQuant packed GPU operators, common HF adaptation';r['kernel_source_sha256']=sha(Path(__file__).with_name('calib_packed_bank.py'))
    elif v['method'].startswith('calib'):
     r['kernel_variant']='CalibQuant materialized numerical port';r['kernel_source_sha256']=sha(Path(__file__).with_name('calib_post.py' if v['method']=='calibpost' else 'calib_bank.py'))
    elif v['method'] in ('rate','affine_rate','anchor_affine'):
     r['kernel_variant']='affine postscale FP32-offset packed reader' if v.get('postscale_reader') else 'mixed native K/V widths and K residual1'
     r['kernel_source_sha256']=sha(Path(__file__).with_name('fused_postscale.py' if v.get('postscale_reader') else 'fused_rates.py'))
    elif v['method']=='hybrid':
     r['kernel_variant']='materialized hybrid quality port';r['kernel_source_sha256']=sha(Path(__file__).with_name('hybrid_bank.py'))
    elif v['method']=='hybrid_packed':
     r['kernel_variant']='packed hybrid rotated K2 and clipped V1';r['kernel_source_sha256']=sha(Path(__file__).with_name('fused_hybrid.py'))
    log.write(json.dumps(r)+'\n');log.flush();n+=1
    print(json.dumps({'done':n,'id':row['id'],'variant':v['name'],'nv':r['visual_tokens'],'seconds':round(time.perf_counter()-t,2)}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','records':n,'expected':len(rows)*len(c['variants'])}))
if __name__=='__main__':main()
