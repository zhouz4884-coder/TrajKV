"""Prospectively fixed new Doc128 comparisons; no selection on test outcomes."""
import csv,hashlib,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm
from audit_book_records import audit

def main():
 frozen=json.loads((ROOT/'audit/a2/book_confirmation_v1.json').read_text(encoding='utf-8'));out=ROOT/'results/book_confirm_doc128_analysis'
 runs=['book_confirm_doc_s0_v1','book_confirm_doc_s1_v1'];sample='configs/a2/docvqa_future128.json';reference='configs/data/docvqa_future128_references.jsonl'
 for path in [sample,reference]:assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==frozen['input_sha256'][path]
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',*runs,'--samples',str(ROOT/sample),'--references',str(ROOT/reference),
  '--expected-variants',str(len(frozen['variants'])),'--role','confirmation','--output',str(out)],check=True,stdout=subprocess.DEVNULL)
 basic=json.loads((out/'summary.json').read_text());assert basic['sessions']==frozen['doc_free_sessions'] and basic['questions_per_variant']==501
 records,audit_result=audit(frozen,runs,frozen['variants'],frozen['input_sha256'][sample]);assert len(records)==basic['sessions']
 by=defaultdict(dict)
 for row in csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')):by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 comparisons=[{'left':left,'right':right,**paired(by[left],by[right],repeats=8000)} for left,right in frozen['primary_superiority_pairs_holm']]
 for row,pv in zip(comparisons,holm([r['p'] for r in comparisons])):row['holm_primary_family']=pv
 candidate=frozen['candidate'];table={row['variant']:row for row in basic['table']};target={'candidate':candidate,'reference':frozen['quality_reference'],
  'budget_bits':table[candidate]['budget_bits_including_encoder_constants'],**paired(by[candidate],by[frozen['quality_reference']],repeats=8000)}
 target.update(point_target_met=target['budget_bits']<=1.8 and target['delta']>=-.005,noninferiority_95_ci=target['budget_bits']<=1.8 and target['ci_low']>=-.005)
 interactions=[]
 for condition in ['cold','warm']:
  both='decoder_'+condition+'_joint';k=both+'_konly';v=both+'_vonly';base='fwht_global'
  contrast={key:by[both][key]-by[k][key]-by[v][key]+by[base][key] for key in by[base]}
  interactions.append({'condition':condition,'scope':'secondary paired ANLS difference-of-differences, not a new method or scored answer',**paired(contrast,{key:0. for key in contrast},repeats=8000)})
 write_csv(out/'primary_comparisons.csv',comparisons);write_csv(out/'storage_target.csv',[target]);write_csv(out/'factorial_interactions.csv',interactions)
 basic.update(primary_candidate=candidate,primary_comparisons=comparisons,storage_target=target,source_audit=audit_result,factorial_interactions=interactions)
 (out/'summary.json').write_text(json.dumps(basic,indent=2));(ROOT/'audit/a2/BOOK_CONFIRM_DOC128_AUDIT.json').write_text(json.dumps(audit_result,indent=2))
 print(json.dumps({'table':basic['table'],'target':target,'comparisons':comparisons,'interactions':interactions,'audit':audit_result}))

if __name__=='__main__':main()
