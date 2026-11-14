"""One bounded fresh-document check, selected only on complete development."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 marker=ROOT/'audit/a2/metadata_confirmation_v1.json'
 if marker.exists():raise RuntimeError('Metadata confirmation already frozen')
 dev=json.loads((ROOT/'results/metadata_development64_analysis/summary.json').read_text());assert dev['metadata_audit']['status']=='PASS' and dev['sessions']==1216
 base=json.loads((ROOT/'configs/a2/metadata_followup_v1_dev_s0.json').read_text());lookup={v['name']:v for v in base['variants']};table={r['variant']:r for r in dev['table']}
 eligible=[r for r in dev['table'] if lookup[r['variant']]['method']=='direction_meta' and r['budget_bits_including_encoder_constants']<=1.8]
 qualified=[r for r in eligible if r['anls']>=table['native4_8']['anls']-.005]
 primary=min(qualified,key=lambda r:(r['budget_bits_including_encoder_constants'],-r['anls'],r['variant'])) if qualified else min(eligible,key=lambda r:(-r['anls'],r['budget_bits_including_encoder_constants'],r['variant']))
 all_group=[r for r in eligible if lookup[r['variant']].get('v_group_size')=='all'];secondary=min(all_group,key=lambda r:(-r['anls'],r['budget_bits_including_encoder_constants'],r['variant']))
 names=list(dict.fromkeys(['full8','native4_8','base2_8','fwht_global','decoder_warm_joint','metadata_identity_all',primary['variant'],secondary['variant']]))
 variants=[lookup[n] for n in names];input_doc='configs/a2/docvqa_meta128.json';refs='configs/data/docvqa_meta128_references.jsonl'
 sample=json.loads((ROOT/input_doc).read_text(encoding='utf-8'));assert len(sample)==128 and sum(r['turns'] for r in sample)==485
 assert json.loads((ROOT/'audit/a2/meta_doc128_transport.json').read_text())['status']=='PASS'
 outputs=[];jobs=[]
 for kind,count,samples in [('doc',64,input_doc),('text',128,'configs/a2/textvqa_future256.json')]:
  for shard in [0,1]:
   name=f'metadata_confirm_{kind}_s{shard}_v1';config=copy.deepcopy(base)
   selected=variants if kind=='doc' else [lookup[n] for n in dict.fromkeys([primary['variant'],secondary['variant']])]
   config.update(start=count*shard,count=count,variants=selected);config['arguments'].update(samples=REMOTE_ROOT+'/'+samples,turns=6 if kind=='doc' else 1,max_patches=8)
   cp=ROOT/'configs/a2'/f'{name}.json';outputs.append((cp,config))
   dependency='book_timing_gpu0_v1' if kind=='doc' else f'metadata_confirm_doc_s{shard}_v1'
   wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
    'settle_seconds':3,'entrypoint':'run_representation_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
   wp=ROOT/'configs/a2'/f'{name}_wait.json';outputs.append((wp,wait))
   job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
    'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
   outputs.append((ROOT/'configs/jobs'/f'{name}.json',job));jobs.append(name)
 assert all(not path.exists() for path,data in outputs)
 for path,data in outputs:path.write_text(json.dumps(data,indent=2),encoding='utf-8')
 old=json.loads((ROOT/'audit/a2/book_confirmation_v1.json').read_text(encoding='utf-8'));sources=set(old['inference_source_sha256'])|{'metadata_direction_bank.py','run_representation_variants.py'}
 external={v[key]:sha(ROOT/v[key].split('/VLAtest/TrajKV/',1)[1]) for v in variants for key in ['direction_book_path','decoder_book_path','metadata_factor_path'] if v.get(key)}
 comparisons=[(primary['variant'],reference) for reference in ['base2_8','decoder_warm_joint','fwht_global']]+[(secondary['variant'],reference) for reference in ['metadata_identity_all',primary['variant'],'base2_8']]
 result={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':primary['variant'],'secondary_low_metadata':secondary['variant'],'selected_development_rows':[primary,secondary],
  'quality_reference':'native4_8','budget_bits':1.8,'quality_margin':.005,'variants':variants,'primary_superiority_pairs_holm':comparisons,
  'selection_rule':'metadata-only family <=1.8bits; cheapest within0.5pp of fixed Native4 if any, otherwise highest complete dev ANLS; secondary highest dev ANLS at whole-image V grouping',
  'doc_source_documents':128,'doc_questions':485,'doc_free_sessions':128*len(variants),'text_selected_methods':list(dict.fromkeys([primary['variant'],secondary['variant']])),
  'text_scope':'transfer reuse of the already evaluated256-image cohort, not an independent new-image confirmation; candidate selected only on Doc development',
  'text_free_answers':256*len(set([primary['variant'],secondary['variant']])),'jobs':jobs,'development_summary_sha256':sha(ROOT/'results/metadata_development64_analysis/summary.json'),
  'inference_source_sha256':{name:sha(ROOT/'code/a2'/name) for name in sorted(sources)},'external_codec_inputs_sha256':external,
  'input_sha256':{path:sha(ROOT/path) for path in [input_doc,refs,'configs/a2/textvqa_future256.json','configs/data/textvqa_future256_references.json']},
  'job_config_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path,data in outputs},'new_doc_model_outcomes_seen':False,
  'scope':'bounded exploration follow-up then all-project summary and public GitHub delivery; full5000x8 stays deferred'}
 marker.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'primary':result['candidate'],'secondary':result['secondary_low_metadata'],'jobs':jobs,'doc_sessions':result['doc_free_sessions'],'text_answers':result['text_free_answers']}))

if __name__=='__main__':main()
