"""Frozen contrasts, deployment costs and source checks for next Doc256."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm

def main():
 frozen=json.loads((ROOT/'audit/a2/next_confirmation_v1.json').read_text());aliases=frozen['aliases'];runs=['next_confirm256_s0_v1','next_confirm256_s1_v1']
 out=ROOT/'results/next_confirm256_analysis';out.mkdir(exist_ok=True,parents=True)
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',*runs,'--samples',str(ROOT/'configs/a2/docvqa_next_confirm256.json'),
  '--expected-variants',str(len(frozen['variants'])),'--output',str(out),'--read-all',aliases['newread_all'],'--role','confirmation'],check=True,stdout=subprocess.DEVNULL)
 basic=json.loads((out/'summary.json').read_text());assert basic['sessions']==frozen['expected_doc_sessions'] and basic['images']==256
 scores=list(csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')));by=defaultdict(dict)
 for row in scores:by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 tests=[]
 for left,right in frozen['primary_superiority_tests']:
  tests.append({'left_label':left,'right_label':right,'left_variant':aliases[left],'right_variant':aliases[right],**paired(by[aliases[left]],by[aliases[right]],repeats=8000)})
 adjusted=holm([r['p'] for r in tests])
 for row,pv in zip(tests,adjusted):row['holm_primary_family']=pv
 reference=frozen['quality_reference_alias'];quality=paired(by['new_storage_primary'],by[reference],repeats=8000);table={r['variant']:r for r in basic['table']}
 bits=table['new_storage_primary']['budget_bits_including_encoder_constants']
 target={'candidate':'new_storage_primary','reference':reference,'budget_bits_including_encoder_constants':bits,**quality,
  'point_target_met':bits<=1.8 and quality['delta']>=-.005,'noninferiority_95_ci':bits<=1.8 and quality['ci_low']>=-.005}
 source_checks=0;codec_checks=0;formulas=0;records=0
 configs={v['name']:v for v in frozen['variants']};seen=set()
 for run in runs:
  folder=ROOT/'results'/run;m=json.loads((folder/'manifest.json').read_text());assert m['answers_visible_to_runner'] is False
  assert m['samples_sha256']==frozen['input_sha256']['configs/a2/docvqa_next_confirm256.json']
  for name,digest in frozen['inference_source_sha256'].items():assert m['sources']['code/a2/'+name]==digest;source_checks+=1
  assert m['external_codec_inputs']==frozen['external_codec_inputs_sha256'];codec_checks+=len(m['external_codec_inputs'])
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key not in seen;seen.add(key);records+=1
   cfg=configs[r['variant']];assert r['variant_config']==cfg and not r['initial_logits_shared_fp16'] and r['read_ledger_exact'] and r.get('global_extra_budget_ok',True)
   o=r['owned_initial'];nv=r['visual_tokens'];e=28*4*nv*128;method=r['method'];base=res=meta=None
   if method in ('rate','affine_rate','base2','native4'):
    kb=cfg.get('bits_k',2) if method in ('rate','affine_rate') else 4 if method=='native4' else 2
    vb=cfg.get('bits_v',1) if method in ('rate','affine_rate') else kb
    base=e*((kb+vb)/8+8/128);res=meta=0
   elif method=='full':base=4*e;res=meta=0
   elif method=='cached_asym':
    base=2*e*(2/8+4/128);res=e*(2/8+4/128)+e*(1/8+2/128)
    meta=28*4*(nv/cfg['page'])*(4*128+12)+28*4*(nv/cfg['page'])+28*2*8
   elif method=='direction':
    dim=r.get('direction_book_dimension');kb=1 if dim is None else 8/dim;vb=cfg.get('bits_v',2)
    base=e*kb/8+28*4*nv*2+28*4*128*2+65536+(256*dim*2 if dim else 0)
    base+=e*(vb/8+4/(nv if cfg.get('v_group_size')=='all' else 128));res=meta=0
    if cfg.get('query_scale_path'):base+=28*4*128*2
   if base is not None:
    assert o['visual_base']==base and o['visual_residual']==res and o['selector_metadata']==meta;formulas+=1
   assert o['visual_position']==8*nv
   if method not in ('full','kvarn2'):assert not o.get('full_visual_retained') and not o.get('diagnostic_fp16_reference')
 assert seen=={(sid,v['name']) for sid in frozen['doc_ids'] for v in frozen['variants']}
 write_csv(out/'primary_comparisons.csv',tests);write_csv(out/'storage_target.csv',[target])
 audit={'status':'PASS','records':records,'source_checks':source_checks,'codec_input_checks':codec_checks,'independent_owner_formulas':formulas,
  'frozen_contract_sha256':hashlib.sha256((ROOT/'audit/a2/next_confirmation_v1.json').read_bytes()).hexdigest()}
 (ROOT/'audit/a2/NEXT_CONFIRM256_AUDIT.json').write_text(json.dumps(audit,indent=2))
 result={**basic,'primary_comparisons':tests,'storage_target':target,'source_audit':audit,'aliases':aliases,'primary_read_variant':aliases['new_read_primary']}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':'PASS','records':records,'storage_target':target,'primary_tests':tests,'read_targets':basic['read_targets'],'table':basic['table']}))
if __name__=='__main__':main()
