"""Bounded new-source metadata confirmation with fixed candidate identities."""
import csv,hashlib,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm
from audit_book_records import audit

def main():
 frozen=json.loads((ROOT/'audit/a2/metadata_confirmation_v1.json').read_text());out=ROOT/'results/metadata_confirm_doc128_analysis'
 runs=['metadata_confirm_doc_s0_v1','metadata_confirm_doc_s1_v1'];sp='configs/a2/docvqa_meta128.json';rp='configs/data/docvqa_meta128_references.jsonl'
 amendment=ROOT/'audit/a2/metadata_execution_cache_v2.json'
 if amendment.exists():mapping=json.loads(amendment.read_text())['run_mapping'];runs=[mapping.get(name,name) for name in runs]
 for path in [sp,rp]:assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==frozen['input_sha256'][path]
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',*runs,'--samples',str(ROOT/sp),'--references',str(ROOT/rp),
  '--expected-variants',str(len(frozen['variants'])),'--role','confirmation','--output',str(out)],check=True,stdout=subprocess.DEVNULL)
 basic=json.loads((out/'summary.json').read_text());assert basic['sessions']==frozen['doc_free_sessions'] and basic['questions_per_variant']==485
 records,check=audit(frozen,runs,frozen['variants'],frozen['input_sha256'][sp]);assert len(records)==basic['sessions'];by=defaultdict(dict)
 for row in csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')):by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 comparisons=[{'left':left,'right':right,**paired(by[left],by[right],repeats=8000)} for left,right in frozen['primary_superiority_pairs_holm']]
 for row,pv in zip(comparisons,holm([r['p'] for r in comparisons])):row['holm_family']=pv
 table={row['variant']:row for row in basic['table']};candidate=frozen['candidate'];target={'candidate':candidate,'reference':'native4_8',
  'budget_bits':table[candidate]['budget_bits_including_encoder_constants'],**paired(by[candidate],by['native4_8'],repeats=8000)}
 target.update(point_target_met=target['budget_bits']<=1.8 and target['delta']>=-.005,noninferiority_95_ci=target['budget_bits']<=1.8 and target['ci_low']>=-.005)
 secondary=[]
 for variant in dict.fromkeys([candidate,frozen['secondary_low_metadata']]):
  for reference in ['base2_8','decoder_warm_joint','full8','native4_8']:
   comparison=paired(by[variant],by[reference],repeats=8000);saving=1-table[variant]['budget_bits_including_encoder_constants']/table[reference]['budget_bits_including_encoder_constants']
   secondary.append({'candidate':variant,'reference':reference,'analysis_role':'secondary resource comparison','visual_byte_saving_including_encoder_constants':saving,**comparison})
 write_csv(out/'primary_comparisons.csv',comparisons);write_csv(out/'storage_target.csv',[target]);write_csv(out/'resource_comparisons.csv',secondary)
 basic.update(primary_candidate=candidate,secondary_low_metadata=frozen['secondary_low_metadata'],primary_comparisons=comparisons,storage_target=target,resource_comparisons=secondary,audit=check)
 (out/'summary.json').write_text(json.dumps(basic,indent=2));(ROOT/'audit/a2/METADATA_CONFIRM_DOC128_AUDIT.json').write_text(json.dumps(check,indent=2))
 print(json.dumps({'table':basic['table'],'target':target,'comparisons':comparisons,'secondary':secondary,'audit':check}))

if __name__=='__main__':main()
