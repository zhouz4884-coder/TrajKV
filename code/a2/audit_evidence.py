"""Independent source preservation, owner formulas, and paired-input audit."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--check-first-cycle',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 errors=[];checks=0;counts={};owner_cases=0
 if a.check_first_cycle:
  old=json.loads((ROOT/'audit/final_artifact_manifest.json').read_text(encoding='utf-8'))['files']
  for name,value in old.items():
   f=ROOT/name;checks+=1
   if not f.exists() or sha(f)!=value['sha256']:errors.append({'kind':'first_cycle_changed','file':name})
 for name in a.runs:
  folder=ROOT/'results'/name;rec=folder/'complete.json'
  if not rec.exists() or json.loads(rec.read_text()).get('status')!='COMPLETE':errors.append({'kind':'incomplete','run':name});continue
  n=0;seen=set()
  for f in folder.rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line);n+=1;key=(r['session_id'],r['variant'])
    if key in seen:errors.append({'kind':'duplicate','run':name,'key':key})
    seen.add(key)
    if r.get('read_ledger_exact') is not True:errors.append({'kind':'read_ledger','run':name,'key':key})
    if r['method'] not in ('rate','base1','base2','native4'):continue
    cfg=r['variant_config'];o=r['owned_initial'];nv=r['visual_tokens'];layers=28;heads=4;dim=128;elems=heads*nv*dim
    if r['method']=='rate' and (cfg.get('kr_bits',0) or cfg.get('v_layers')):continue
    expected=0
    for axis in ('k','v'):
     bits=cfg.get('bits_'+axis,1) if r['method']=='rate' else (1 if r['method']=='base1' else 4 if r['method']=='native4' else 2)
     widths=cfg.get('layer_bits_'+axis,[bits]*layers)
     group=cfg.get(axis+'_group_size',128)
     group=(nv if axis=='k' or cfg.get('v_layout','token')=='channel' else dim) if group=='all' else group
     expected+=sum(elems*b/8+4*elems/group for b in widths)
    if r['method']=='rate' and cfg.get('k_token_gain'):expected+=layers*nv*2*(1 if cfg.get('gain_shared') else heads)
    owner_cases+=1
    if expected!=o['visual_base']:errors.append({'kind':'owner_formula','run':name,'key':key,'expected':expected,'observed':o['visual_base']})
    if o['visual_position']!=8*nv:errors.append({'kind':'shared_visual_position','run':name,'key':key})
    if o.get('diagnostic_fp16_reference') or o.get('full_visual_retained'):errors.append({'kind':'unexpected_reference_owner','run':name,'key':key})
  counts[name]=n
 result={'status':'PASS' if not errors else 'FAIL','first_cycle_files_verified':checks,'owner_formula_cases':owner_cases,'records_per_run':counts,'errors':errors}
 a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if errors:raise RuntimeError('Evidence audit failed')
if __name__=='__main__':main()
