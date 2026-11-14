"""Frozen identity, exact coverage and independent tensor-owner formulas."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 frozen=json.loads((ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json').read_text());expected={(s,v['name']) for s in frozen['confirm_ids'] for v in frozen['variants']};config={v['name']:v for v in frozen['variants']}
 errors=[];seen=set();sources=0;formulas=0;rows=0
 def check(ok,kind,detail):
  if not ok:errors.append({'kind':kind,'detail':detail})
 for run in a.runs:
  folder=ROOT/'results'/run;receipt=folder/'complete.json'
  check(receipt.exists() and json.loads(receipt.read_text()).get('status')=='COMPLETE','receipt',run)
  for mf in folder.rglob('manifest.json'):
   m=json.loads(mf.read_text());ss=m.get('sources')
   if not ss:continue
   for name,h in frozen['source_sha256'].items():
    check(ss.get('code/a2/'+name)==h,'frozen_source',run+':'+name);sources+=1
   check(m['samples_sha256']==frozen['confirm_membership_sha256'],'sample_identity',run)
   for name,h in frozen['external_codec_inputs_sha256'].items():check(m.get('external_codec_inputs',{}).get(name)==h,'external_codec_input',run+':'+name)
  for f in folder.rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line);rows+=1;key=(r['session_id'],r['variant']);check(key not in seen,'duplicate',str(key));seen.add(key)
    check(key in expected,'unexpected',str(key));cfg=config[r['variant']]
    check(r['variant_config']==cfg,'variant_config',str(key));check(r.get('initial_logits_shared_fp16') is False,'compression_point',str(key));check(r['read_ledger_exact'],'ledger',str(key))
    check(r.get('read_budget_ok',True) and r.get('global_extra_budget_ok',True),'budget',str(key))
    o=r['owned_initial'];nv=r['visual_tokens'];layers=28;heads=4;d=128;e=layers*heads*nv*d;method=r['method'];base=res=meta=None
    if method in ('base1','base2','native4','rate'):
     widths=(1 if method=='base1' else 2 if method=='base2' else 4) if method!='rate' else cfg.get('bits_k',1)
     base=0
     for axis,bits in [('k',widths),('v',cfg.get('bits_v',1) if method=='rate' else widths)]:
      gs=cfg.get(axis+'_group_size',128);gs=(nv if axis=='k' or cfg.get('v_layout','token')=='channel' else d) if gs=='all' else gs
      base+=sum(heads*nv*d*(b/8+4/gs) for b in cfg.get('layer_bits_'+axis,[bits]*layers))
     if cfg.get('k_token_gain'):base+=layers*nv*2*(1 if cfg.get('gain_shared') else heads)
     res=0;meta=0
    elif method=='asym':
     base=2*e*(2/8+4/128);res=e*(2/8+4/128)+e*(1/8+2/128);meta=layers*heads*(nv/cfg.get('page',256))*(4*d+12)
    elif method=='full':base=4*e;res=0;meta=0
    elif method=='direction':
     dim=r.get('direction_book_dimension');kb=1 if dim is None else 8/dim;vb=cfg.get('bits_v',2);vg=nv if cfg.get('v_group_size')=='all' else 128
     base=e*kb/8+layers*heads*nv*2+layers*heads*d*2+65536
     if dim is not None:base+=256*dim*2
     if cfg.get('query_scale_path'):base+=layers*heads*d*2
     base+=2*e if vb==16 else e*(vb/8+4/vg);res=0;meta=0
    elif method=='calib_packed':
     kb=cfg.get('bits_k',2);vb=cfg.get('bits_v',2);base=e*(kb+vb)/8+8*layers*heads*d+sum((1<<b)*2 for b in {kb,vb});res=0;meta=0
    if base is not None:
     formulas+=1;check(base==o['visual_base'],'visual_base_formula',{'key':key,'expected':base,'observed':o['visual_base']})
     check(res==o['visual_residual'],'residual_formula',{'key':key,'expected':res,'observed':o['visual_residual']});check(meta==o.get('selector_metadata',0),'metadata_formula',{'key':key,'expected':meta,'observed':o.get('selector_metadata')})
    check(o['visual_position']==8*nv,'shared_position_owner',str(key))
    if method not in ('oraclek','full','kvarn2'):
     check(not o.get('diagnostic_fp16_reference') and not o.get('full_visual_retained'),'unrequested_fp16_reference',str(key))
 check(seen==expected,'coverage',{'expected':len(expected),'seen':len(seen)})
 result={'status':'PASS' if not errors else 'FAIL','records':rows,'expected_records':len(expected),'frozen_source_comparisons':sources,'independent_owner_formula_cases':formulas,'errors':errors}
 a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if errors:raise RuntimeError('Frozen confirmation audit failed')
if __name__=='__main__':main()
