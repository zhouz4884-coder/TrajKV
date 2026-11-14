"""Combine unique real development records and verify repeated controls."""
import hashlib,json,subprocess,sys
import csv
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv
from analyze_docvqa_confirm64 import holm

def main():
 if (ROOT/'audit/a2/book_confirmation_v1.json').exists():raise RuntimeError('Development summary is a frozen selection input; add later statistics in separate files')
 base_runs=['book_development64_s0_v2','book_development64_s1_v2'];extra_runs=['book_factorial64_s0_v1','book_factorial64_s1_v1']
 base_config=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text(encoding='utf-8'))
 extra_config=json.loads((ROOT/'configs/a2/book_factorial64_s0_v1.json').read_text(encoding='utf-8'))
 variants={v['name']:v for v in base_config['variants']}
 for v in extra_config['variants']:
  if v['name'] in variants:assert variants[v['name']]==v
  else:variants[v['name']]=v
 assert len(variants)==15;unique={};origins={};duplicates=[];raw_count=0;owner_checks=0
 for run in base_runs+extra_runs:
  folder=ROOT/'results'/run;assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE'
  path=folder/'quality/records.jsonl' if run in base_runs else folder/'records.jsonl'
  manifest=json.loads((path.parent/'manifest.json').read_text());assert manifest['answers_visible_to_runner'] is False
  if run in extra_runs:
   for name,digest in json.loads((ROOT/'audit/a2/book_factorial64_design_v1.json').read_text())['source_sha256'].items():assert manifest['sources']['code/a2/'+name]==digest
  for line in path.read_text().splitlines():
   r=json.loads(line);raw_count+=1;key=(r['session_id'],r['variant']);assert r['variant_config']==variants[r['variant']]
   if key in unique:
    previous=json.loads(unique[key]);assert previous['owned_initial']==r['owned_initial']
    assert [t['token_ids'] for t in previous['turns']]==[t['token_ids'] for t in r['turns']]
    duplicates.append({'session_id':key[0],'variant':key[1],'initial_logits_identical':previous['initial_logits_sha256']==r['initial_logits_sha256']});continue
   unique[key]=line;origins['|'.join(key)]={'run':run,'source_record_sha256':hashlib.sha256(line.encode()).hexdigest()}
   if r['method']=='learned_direction':
    nv=r['visual_tokens'];e=28*4*nv*128;expected=e*(2/8+1/8+4/128)+28*4*nv*2+28*4*128*2+2048+r['decoder_book_bytes'];owned=r['owned_initial']
    assert owned['visual_base']==expected and owned['visual_residual']==0 and owned['selector_metadata']==0 and owned['visual_position']==8*nv
    assert not owned.get('full_visual_retained') and not owned.get('diagnostic_fp16_reference')
    assert 16*(expected+8*nv+r.get('encoder_calibration_parameters_bytes',0))/(4*e)<=1.8+1e-10
    if r['variant'].endswith('_vonly'):assert r['decoder_book_bytes']==0 and r['encoder_calibration_parameters_bytes']==896
    if r['variant'].endswith('_konly'):assert r['encoder_calibration_parameters_bytes']==0
    owner_checks+=1
 assert len(unique)==960 and len(duplicates)==192 and raw_count==1152
 union_name='book_development64_analysis_union';union=ROOT/'results'/union_name;union.mkdir(exist_ok=True,parents=True)
 config=dict(base_config,start=0,count=64,variants=list(variants.values()))
 sample=ROOT/'configs/a2/docvqa_next_dev64.json'
 (union/'manifest.json').write_text(json.dumps({'config':config,'samples_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),
  'scope':'analysis-only union of unique real records from four immutable GPU jobs; repeated controls retained separately, no synthetic inference',
  'raw_runs':base_runs+extra_runs,'task_inference_performed_by_this_union':False},indent=2))
 (union/'records.jsonl').write_text(''.join(line+'\n' for line in unique.values()))
 (union/'origins.json').write_text(json.dumps(origins,indent=2));(union/'complete.json').write_text(json.dumps({'status':'COMPLETE','unique_real_records':len(unique),'raw_gpu_sessions':raw_count,'analysis_only':True}))
 out=ROOT/'results/book_factorial64_analysis'
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',union_name,'--samples',str(sample),'--expected-variants','15','--output',str(out)],check=True,stdout=subprocess.DEVNULL)
 summary=json.loads((out/'summary.json').read_text());audit={'status':'PASS','raw_gpu_sessions':raw_count,'unique_real_sessions':len(unique),'repeated_control_pairs':len(duplicates),
  'repeated_control_token_trajectories_identical':len(duplicates),'repeated_control_first_logits_identical':sum(r['initial_logits_identical'] for r in duplicates),'learned_owner_formula_cases':owner_checks}
 by=defaultdict(dict)
 for row in csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')):by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 comparisons=[];interactions=[]
 for condition in ['cold','warm']:
  both='decoder_'+condition+'_joint';k=both+'_konly';v=both+'_vonly';base='fwht_global'
  for left,right in [(both,base),(k,base),(v,base),(both,k),(both,v)]:comparisons.append({'condition':condition,'left':left,'right':right,**paired(by[left],by[right],repeats=8000)})
  contrast={key:by[both][key]-by[k][key]-by[v][key]+by[base][key] for key in by[base]}
  interactions.append({'condition':condition,'analysis_role':'open development factorial difference-of-differences',**paired(contrast,{key:0. for key in contrast},repeats=8000)})
 for row,pv in zip(comparisons,holm([r['p'] for r in comparisons])):row['holm_exploratory_family']=pv
 write_csv(out/'factorial_comparisons.csv',comparisons);write_csv(out/'factorial_interactions.csv',interactions)
 summary.update(factorial_audit=audit,book_audit=json.loads((ROOT/'audit/a2/BOOK_DEVELOPMENT64_AUDIT.json').read_text()),raw_gpu_sessions=raw_count,factorial_comparisons=comparisons,factorial_interactions=interactions)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'repeated_controls.json').write_text(json.dumps(duplicates,indent=2))
 (ROOT/'audit/a2/BOOK_FACTORIAL64_AUDIT.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps({'audit':audit,'table':summary['table']}))

if __name__=='__main__':main()
