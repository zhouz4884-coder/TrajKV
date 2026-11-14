"""Complete metadata development and exact old/new implementation controls."""
import csv,hashlib,json,subprocess,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze import paired,write_csv

def main():
 if (ROOT/'audit/a2/metadata_confirmation_v1.json').exists():raise RuntimeError('Development selection input is frozen; do not overwrite')
 root=ROOT/'results/metadata_followup_v1';assert json.loads((root/'complete.json').read_text())['status']=='COMPLETE'
 runs=['metadata_followup_v1/development_s0','metadata_followup_v1/development_s1'];out=ROOT/'results/metadata_development64_analysis'
 subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_next_docvqa.py')),'--runs',*runs,'--samples',str(ROOT/'configs/a2/docvqa_next_dev64.json'),
  '--expected-variants','19','--output',str(out)],check=True,stdout=subprocess.DEVNULL)
 data=json.loads((out/'summary.json').read_text());assert data['sessions']==1216;records={};formulas=codec_checks=0
 for run in runs:
  folder=ROOT/'results'/run;manifest=json.loads((folder/'manifest.json').read_text());assert not manifest['answers_visible_to_runner']
  for path,digest in manifest['external_codec_inputs'].items():
   local=ROOT/path.split('/VLAtest/TrajKV/',1)[1];assert hashlib.sha256(local.read_bytes()).hexdigest()==digest;codec_checks+=1
  for line in (folder/'records.jsonl').read_text().splitlines():
   r=json.loads(line);key=(r['session_id'],r['variant']);assert key not in records;records[key]=r
   if r['method']!='direction_meta':continue
   nv=r['visual_tokens'];e=28*4*nv*128;c=r['variant_config'];g=nv if c.get('v_group_size')=='all' else 128
   expected=e*(3/8+4/g)+28*4*nv*2+28*4*128*2+2048;o=r['owned_initial'];encoder=0
   assert o['visual_base']==expected and o['visual_residual']==0 and o['selector_metadata']==0 and o['visual_position']==8*nv and r['decoder_book_bytes']==0
   if c.get('metadata_factor_path'):
    artifact=json.loads((ROOT/c['metadata_factor_path'].split('/VLAtest/TrajKV/',1)[1]).read_text());assert artifact['model_weights_frozen'] and not artifact['labels_used'] and artifact['steps']==400
    encoder=artifact['parameter_count']*4
   assert r['encoder_calibration_parameters_bytes']==encoder and r['value_group_size']==g
   assert 16*(expected+8*nv+encoder)/(4*e)<=1.8 and not o.get('full_visual_retained') and not o.get('diagnostic_fp16_reference');formulas+=1
 old={}
 for run in ['book_development64_s0_v2','book_development64_s1_v2']:
  for line in (ROOT/'results'/run/'quality/records.jsonl').read_text().splitlines():
   r=json.loads(line);old[r['session_id'],r['variant']]=r
 ids={key[0] for key in records};repeats=identity=0
 for sid in ids:
  for variant in ['full8','native4_8','base2_8','fwht_global','decoder_warm_joint']:
   a=old[sid,variant];b=records[sid,variant]
   assert a['initial_logits_sha256']==b['initial_logits_sha256'] and a['owned_initial']==b['owned_initial']
   assert [t['token_ids'] for t in a['turns']]==[t['token_ids'] for t in b['turns']];repeats+=1
  a=records[sid,'fwht_global'];b=records[sid,'metadata_identity_g128']
  assert a['initial_logits_sha256']==b['initial_logits_sha256'] and a['owned_initial']==b['owned_initial']
  assert [t['token_ids'] for t in a['turns']]==[t['token_ids'] for t in b['turns']];identity+=1
 by=defaultdict(dict)
 for row in csv.DictReader((out/'scores.csv').open(encoding='utf-8-sig')):by[row['variant']][row['image_id'],int(row['turn'])]=float(row['score'])
 comparisons=[]
 for variant in sorted(by):
  if '_meta_' not in variant:continue
  baseline='metadata_identity_all' if '_all_' in variant else 'fwht_global'
  for reference in [baseline,'base2_8','decoder_warm_joint','native4_8']:comparisons.append({'variant':variant,'reference':reference,'role':'open development',**paired(by[variant],by[reference],repeats=8000)})
 write_csv(out/'metadata_comparisons.csv',comparisons)
 audit={'status':'PASS','free_sessions':len(records),'metadata_owner_formula_cases':formulas,'external_codec_checks':codec_checks,
  'old_vs_new_identical_trajectories_logits_owners':repeats,'metadata_identity_equals_original_global':identity}
 data.update(metadata_audit=audit,metadata_comparisons=comparisons);(out/'summary.json').write_text(json.dumps(data,indent=2));(ROOT/'audit/a2/METADATA_DEVELOPMENT64_AUDIT.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps({'audit':audit,'table':data['table']}))

if __name__=='__main__':main()
