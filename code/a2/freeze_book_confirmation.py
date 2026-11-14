"""Select on complete Doc dev64, freeze new source-disjoint Doc/Text tests."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 marker=ROOT/'audit/a2/book_confirmation_v1.json'
 if marker.exists():raise RuntimeError('Book confirmation already frozen')
 dev=json.loads((ROOT/'results/book_factorial64_analysis/summary.json').read_text(encoding='utf-8'))
 assert dev['sessions']==960 and dev['book_audit']['status']==dev['factorial_audit']['status']=='PASS'
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text(encoding='utf-8'));variants=base['variants']
 additional=json.loads((ROOT/'configs/a2/book_factorial64_s0_v1.json').read_text(encoding='utf-8'))['variants']
 variants += [v for v in additional if v['name'] not in {row['name'] for row in variants}]
 # Two already-fixed official-operator adaptations remain comparison controls;
 # no candidate is selected using their future outcomes.
 historical=json.loads((ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json').read_text())
 variants += [v for v in historical['variants'] if v['name'] in ['calib2_fp32','calib_k2v1_fp32']]
 calib2=next(v for v in variants if v['name']=='calib2_fp32');variants.append(dict(calib2,name='calib2_fp32_tau0',tau=0.0))
 base['variants']=variants;lookup={v['name']:v for v in variants};assert len(variants)==18
 reference=next(r['anls'] for r in dev['table'] if r['variant']=='native4_8')
 eligible=[r for r in dev['table'] if r['role']=='actual packed runtime' and r['budget_bits_including_encoder_constants']<=1.8]
 qualified=[r for r in eligible if r['anls']>=reference-.005]
 chosen=min(qualified,key=lambda r:(r['budget_bits_including_encoder_constants'],-r['anls'],r['variant'])) if qualified else min(eligible,key=lambda r:(-r['anls'],r['budget_bits_including_encoder_constants'],r['variant']))
 candidate=chosen['variant'];pairs=[(candidate,r) for r in ['direction_vq2_v1','k2_lloyd_v1','base2_8','fwht_untrained_headwise']]
 pairs += [('decoder_cold_book_only','decoder_warm_book_only'),('decoder_cold_joint','decoder_warm_joint'),('decoder_cold_joint','decoder_cold_book_only'),
  ('decoder_warm_joint','decoder_warm_book_only'),('fwht_global','direction_vq2_v1')]
 pairs += [('decoder_warm_joint','decoder_warm_joint_vonly'),('decoder_cold_joint','decoder_cold_joint_vonly'),
  ('decoder_warm_joint_vonly','fwht_global'),('decoder_cold_joint_vonly','fwht_global'),(candidate,'calib_k2v1_fp32'),(candidate,'calib2_fp32')]
 inputs=['configs/a2/docvqa_future128.json','configs/data/docvqa_future128_references.jsonl','configs/a2/textvqa_future256.json','configs/data/textvqa_future256_references.json']
 doc=json.loads((ROOT/inputs[0]).read_text(encoding='utf-8'));text=json.loads((ROOT/inputs[2]).read_text(encoding='utf-8'));assert len(doc)==128 and len(text)==256
 assert len({row['source_document_id'] for row in doc})==128 and sum(row['turns'] for row in doc)==501
 for path in ['audit/a2/future_doc128_transport.json','audit/a2/textvqa_future256_transport.json']:assert json.loads((ROOT/path).read_text())['status']=='PASS'
 old=json.loads((ROOT/'audit/a2/next_confirmation_v1.json').read_text());names=set(old['inference_source_sha256'])|{'fast_wht.py','fused_learned_direction.py','learned_direction_bank.py'}
 sources={name:sha(ROOT/'code/a2'/name) for name in sorted(names)};external={}
 for variant in variants:
  for key in ['direction_book_path','decoder_book_path']:
   if variant.get(key):external[variant[key]]=sha(ROOT/variant[key].split('/VLAtest/TrajKV/',1)[1])
 outputs=[];jobs=[]
 for kind,count,samples in [('doc',64,inputs[0]),('text',128,inputs[2])]:
  for shard in [0,1]:
   name=f'book_confirm_{kind}_s{shard}_v1';config=copy.deepcopy(base);config.update(start=count*shard,count=count)
   config['arguments'].update(samples=REMOTE_ROOT+'/'+samples,turns=6 if kind=='doc' else 1,max_patches=8)
   cp=ROOT/'configs/a2'/f'{name}.json';outputs.append((cp,config))
   dependency=f'book_geometry16_s{shard}_v1' if kind=='doc' else f'book_confirm_doc_s{shard}_v1'
   wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
    'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
   wp=ROOT/'configs/a2'/f'{name}_wait.json';outputs.append((wp,wait))
   job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
    'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
   outputs.append((ROOT/'configs/jobs'/f'{name}.json',job));jobs.append(name)
 assert all(not path.exists() for path,data in outputs)
 for path,data in outputs:path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
 result={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':candidate,'development_selected_row':chosen,'development_quality_reference':'native4_8',
  'selection_rule':'among actual packed candidates <=1.8bits, choose cheapest within0.5pp of fixed Native4 reference; otherwise highest complete dev ANLS, break ties by actual bytes then name',
  'development_target_met':bool(qualified),'variants':variants,'quality_reference':'native4_8','quality_margin':.005,'budget_bits':1.8,
  'doc_source_documents':128,'doc_questions_per_variant':501,'doc_free_sessions':128*len(variants),'text_images':256,'text_strata':{'numeric':128,'text':128},
  'text_free_answers':256*len(variants),'primary_superiority_pairs_holm':pairs,'primary_family_size':len(pairs),
  'identity_comparison_rule':'retain any identical-method contrast with p=1; do not remove a frozen candidate or revise family after test outcomes',
  'primary_ni_rule':'candidate vs Native4, point margin0.5pp and paired source-document95%CI noninferiority reported separately; budget includes encoder constants',
  'official_operator_controls':'same frozen CalibQuant quant/encode/QK/WV source, HF visual-only adaptation; not original model/serving reproduction; different actual byte prices remain visible',
  'transfer_rule':'TextVQA official10-annotator soft accuracy; balanced within-subset transfer, not official overall distribution; no selection using Text outcomes',
  'inference_source_sha256':sources,'external_codec_inputs_sha256':external,'input_sha256':{path:sha(ROOT/path) for path in inputs},
  'job_config_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path,data in outputs},'jobs':jobs,
  'development_summary_sha256':sha(ROOT/'results/book_factorial64_analysis/summary.json'),
  'source_disjoint_membership_sha256':{path:sha(ROOT/path) for path in ['audit/a2/future_doc128_membership.json','audit/a2/textvqa_future256_membership.json']},
  'new_test_model_outcomes_seen':False,'scope':'new data confirmation of fixed candidates; old Doc256/Text128 are already observed and not relabeled as blind'}
 marker.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'candidate':candidate,'development_target_met':bool(qualified),'jobs':jobs,'doc_sessions':result['doc_free_sessions'],'text_answers':result['text_free_answers']}))

if __name__=='__main__':main()
