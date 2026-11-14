"""Completed packed-codebook development: paired ablations and owner audit."""
import csv,hashlib,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm
from costs import visual_owner

def main():
 runs=['book_development64_s0_v2','book_development64_s1_v2'];out=ROOT/'results/book_development64_analysis';out.mkdir(exist_ok=True,parents=True)
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',*runs,'--samples',str(ROOT/'configs/a2/docvqa_next_dev64.json'),
  '--expected-variants','11','--output',str(out)],check=True,stdout=subprocess.DEVNULL)
 summary=json.loads((out/'summary.json').read_text());assert summary['sessions']==704
 scores=list(csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')));by=defaultdict(dict)
 for row in scores:by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 pairs=[('fwht_global','direction_vq2_v1'),('fwht_untrained_headwise','fwht_global'),('decoder_cold_book_only','decoder_warm_book_only'),
  ('decoder_cold_joint','decoder_warm_joint'),('decoder_cold_joint','decoder_cold_book_only'),('decoder_warm_joint','decoder_warm_book_only')]
 for variant in ['decoder_cold_book_only','decoder_warm_book_only','decoder_cold_joint','decoder_warm_joint']:
  pairs += [(variant,reference) for reference in ['fwht_untrained_headwise','direction_vq2_v1','k2_lloyd_v1','base2_8','native4_8']]
 comparisons=[{'left':left,'right':right,'analysis_role':'open development',**paired(by[left],by[right],repeats=8000)} for left,right in pairs]
 for row,adjusted in zip(comparisons,holm([r['p'] for r in comparisons])):row['holm_exploratory_family']=adjusted
 records={};owner_checks=codec_checks=source_checks=0
 for run in runs:
  folder=ROOT/'results'/run/'quality';manifest=json.loads((folder/'manifest.json').read_text());assert not manifest['answers_visible_to_runner']
  for path,digest in manifest['external_codec_inputs'].items():
   local=ROOT/path.split('/VLAtest/TrajKV/',1)[1];assert hashlib.sha256(local.read_bytes()).hexdigest()==digest;codec_checks+=1
  for task_shard in [0,1]:
   for task in ['book_only','joint']:
    fit=json.loads((ROOT/f'results/next_books_s{task_shard}_v2/{task}/manifest.json').read_text())
    for source in ['fused_learned_direction.py','fast_wht.py']:
     assert manifest['sources']['code/a2/'+source]==fit['sources'][source];source_checks+=1
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key not in records;records[key]=r
   if r['method']!='learned_direction':continue
   nv=r['visual_tokens'];e=28*4*nv*128;o=r['owned_initial'];headwise=r['decoder_headwise']
   expected=e*(2/8+1/8+4/128)+28*4*nv*2+28*4*128*2+2048+(28*4*256*4*2 if headwise else 0)
   assert o['visual_base']==expected and o['visual_residual']==0 and o['selector_metadata']==0 and o['visual_position']==8*nv
   assert not o.get('full_visual_retained') and not o.get('diagnostic_fp16_reference') and not r['rotation_matrix_retained']
   denominator=4*e;bits=16*(visual_owner(r)[0]+r.get('encoder_calibration_parameters_bytes',0))/denominator
   assert bits<=1.8+1e-10;owner_checks+=1
 ids={key[0] for key in records};control_pairs=0
 for sid in ids:
  a=records[sid,'fwht_global'];b=records[sid,'fwht_untrained_headwise']
  assert a['initial_logits_sha256']==b['initial_logits_sha256']
  assert [t['token_ids'] for t in a['turns']]==[t['token_ids'] for t in b['turns']];control_pairs+=1
 write_csv(out/'book_comparisons.csv',comparisons)
 audit={'status':'PASS','records':len(records),'learned_format_owner_formula_cases':owner_checks,'same_global_and_repeated_headwise_trajectories':control_pairs,
  'external_codec_checks':codec_checks,'calibration_inference_kernel_identity_checks':source_checks,'every_learned_image_including_encoder_constants_le_1p8':True}
 (ROOT/'audit/a2/BOOK_DEVELOPMENT64_AUDIT.json').write_text(json.dumps(audit,indent=2))
 summary.update(book_comparisons=comparisons,book_audit=audit)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({'table':summary['table'],'comparisons':comparisons,'audit':audit}))

if __name__=='__main__':main()
