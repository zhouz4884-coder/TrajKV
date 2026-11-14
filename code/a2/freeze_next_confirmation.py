"""Freeze next methods only after complete new-source development on both axes."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def key(cfg):return json.dumps({k:v for k,v in cfg.items() if k!='name'},sort_keys=True)
def save(path,value):
 if path.exists():raise RuntimeError('Refuse to overwrite frozen artifact: '+str(path))
 path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
 frozen_path=ROOT/'audit/a2/next_confirmation_v1.json'
 if frozen_path.exists():raise RuntimeError('Next confirmation already frozen')
 for name in ['next_confirm256_s0_v1','next_confirm256_s1_v1','next_textbalanced128_s0_v1','next_textbalanced128_s1_v1']:
  if (ROOT/'results'/name).exists():raise RuntimeError('Next confirmation outcomes already exist')
 dev_path=ROOT/'results/trajectory_development64_analysis/summary.json';read_path=ROOT/'results/cached_reads_factorial64_analysis/summary.json'
 dev=json.loads(dev_path.read_text());reads=json.loads(read_path.read_text());assert dev['sessions']==896 and reads['sessions']==640
 table={r['variant']:r for r in dev['table']};ref_order=['full8','base2_8','native4_8','kvarn2_8_quality_port']
 reference=sorted([table[n] for n in ref_order],key=lambda r:(-r['anls'],ref_order.index(r['variant'])))[0]
 eligible=[r for r in dev['table'] if r['role']=='actual packed runtime' and r['budget_bits_including_encoder_constants']<=1.8]
 qualified=[r for r in eligible if r['anls']>=reference['anls']-.005]
 chosen=sorted(qualified,key=lambda r:(r['budget_bits_including_encoder_constants'],-r['anls'],r['variant']))[0] if qualified else sorted(eligible,key=lambda r:(-r['anls'],r['budget_bits_including_encoder_constants'],r['variant']))[0]
 rt={r['variant']:r for r in reads['table']};targets=reads['read_targets'];rqualified=[r for r in targets if r['point_target_met']]
 safe=[r for r in targets if r['extra_read_saving']>=.5];assert safe
 rchosen=sorted(rqualified if rqualified else safe,key=lambda r:(-rt[r['variant']]['anls'],-r['extra_read_saving'],r['variant']))[0]
 storage_cfg={v['name']:v for v in json.loads((ROOT/'audit/a2/next_development_design_v1.json').read_text())['variants']}
 read_cfg={v['name']:v for v in json.loads((ROOT/'audit/a2/cached_reads_development_v1.json').read_text())['variants']}
 extra_read=json.loads((ROOT/'audit/a2/read_factorial_development_v1.json').read_text())['new_variant'];read_cfg[extra_read['name']]=extra_read
 variants=[];aliases={};by_key={}
 def add(name,cfg):
  cfg=copy.deepcopy(cfg);cfg['name']=name;k=key(cfg)
  if k in by_key:aliases[name]=by_key[k];return
  by_key[k]=name;aliases[name]=name;variants.append(cfg)
 add('new_storage_primary',storage_cfg[chosen['variant']]);aliases[chosen['variant']]='new_storage_primary'
 for name in ['full8','base2_8','native4_8','kvarn2_8_quality_port','k2_lloyd_v1','previous_frozen_primary',
  'cold_first_gain','warm_first_gain','cold_multi_gain','warm_multi_gain','cold_multi_affine','warm_multi_affine']:
  add(name,storage_cfg[name])
 density={v['name']:v for v in json.loads((ROOT/'audit/a2/density_development_design_v1.json').read_text())['variants']}
 for name,source in [('full16','full8_cap16'),('base2_16','base2_cap16'),('native4_16','native4_cap16')]:add(name,density[source])
 add('new_read_primary',read_cfg[rchosen['variant']]);selected_fraction=read_cfg[rchosen['variant']].get('fraction',.25)
 suffix='50' if selected_fraction==.5 else '25'
 for name,source in [('newread_all','cachedread_all'),('newread_none','cachedread_none'),('newread_periodic','cachedread_periodic25'),
  ('newread_freshrisk','cachedread_risk'+suffix),('newread_cached','cachedread_cached'+suffix),('newread_prefill_cached','cachedread_prefill_cached'+suffix),
  ('newread_prefill_fresh','cachedread_prefill_freshrisk50')]:
  add(name,read_cfg[source])
 doc_samples=json.loads((ROOT/'configs/a2/docvqa_next_confirm256.json').read_text(encoding='utf-8'))
 assert len(doc_samples)==256 and len({r['source_document_id'] for r in doc_samples})==256 and sum(len(r['questions']) for r in doc_samples)==999
 text_samples=json.loads((ROOT/'configs/a2/textvqa_balanced128.json').read_text(encoding='utf-8'));assert len(text_samples)==128
 base=json.loads((ROOT/'configs/a2/trajectory_development64_s0_v1.json').read_text());base['variants']=variants
 jobs=[];producer_configs={}
 for shard in (0,1):
  name=f'next_confirm256_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=128*shard,count=128)
  cfg['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/docvqa_next_confirm256.json',turns=6)
  cp=ROOT/'configs/a2'/f'{name}.json';save(cp,cfg);producer_configs[cp.name]=sha(cp)
  dependencies=[{'complete':REMOTE_ROOT+f'/results/read_factorial_dev64_s{s}_v1/complete.json','launch':REMOTE_ROOT+f'/jobs/read_factorial_dev64_s{s}_v1.launch.json'} for s in (0,1)]
  wait={'dependencies':dependencies,'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';save(wp,wait)
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  save(ROOT/'configs/jobs'/f'{name}.json',job);jobs.append(name)
  external=f'next_textbalanced128_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=64*shard,count=64)
  cfg['variants']=[v for v in variants if v['method']!='cached_asym' and v.get('cap',8)==8];cfg['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/textvqa_balanced128.json',turns=1)
  cp=ROOT/'configs/a2'/f'{external}.json';save(cp,cfg);producer_configs[cp.name]=sha(cp)
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{name}/complete.json','launch':REMOTE_ROOT+f'/jobs/{name}.launch.json'}],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{external}_wait.json';save(wp,wait)
  job['name']=external;job['argv']=[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+external]
  save(ROOT/'configs/jobs'/f'{external}.json',job);jobs.append(external)
 codec={}
 for v in variants:
  for field in ['affine_factor_path','direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path']:
   remote=v.get(field)
   if remote:
    assert remote.startswith(REMOTE_ROOT+'/');local=ROOT/remote[len(REMOTE_ROOT)+1:];codec[remote]=sha(local)
 tests=[['cold_multi_gain','warm_multi_gain'],['cold_multi_gain','cold_first_gain'],['cold_multi_affine','cold_multi_gain'],
  ['cold_multi_affine','warm_multi_affine'],['new_storage_primary','k2_lloyd_v1'],['new_storage_primary','previous_frozen_primary'],
  ['new_read_primary','newread_none'],['newread_prefill_cached','newread_cached'],['newread_prefill_fresh','newread_freshrisk'],['newread_prefill_cached','newread_prefill_fresh']]
 result={'schema':'trajkv.next-confirmation.v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'primary_storage_selection':chosen,
  'development_storage_target_met':bool(qualified),'quality_reference_development':reference,'quality_reference_alias':aliases[reference['variant']],
  'primary_read_selection':rchosen,'development_read_target_met':bool(rqualified),'variants':variants,'aliases':aliases,
  'selection_rule':'storage: unaugmented cap8 candidates from the prespecified14-config development; lowest conservative bits<=1.8 within0.5pp of strongest reference, otherwise highest ANLS<=1.8. Read: highest ANLS among >=50% extra-read-saving candidates, ties larger saving. None remains a candidate and explicit control. Generic anchor transfer was not promoted after native-versus-compressed controls showed no precision gain; cap16 density variants remain matched controls rather than changing the storage input setting.',
  'primary_superiority_tests':tests,'statistics':'paired whole-source-document bootstrap8000, ten planned contrasts Holm adjusted; exact and ANLS both reported;0.5pp point and95% noninferiority interval distinct',
  'storage_target':'conservative bits<=1.8 and ANLS delta>=-0.005 to frozen strongest reference; no requirement to beat FP16',
  'read_target':'same owned storage each image, aggregate extra logical read saving>=0.5 and ANLS delta>=-0.005; physical HBM and fixed-token latency separate',
  'doc_ids':[r['id'] for r in doc_samples],'doc_images':256,'doc_questions':999,'expected_doc_sessions':256*len(variants),
  'text_ids':[r['id'] for r in text_samples],'text_variants':[v for v in variants if v['method']!='cached_asym' and v.get('cap',8)==8],
  'expected_text_sessions':128*len([v for v in variants if v['method']!='cached_asym' and v.get('cap',8)==8]),
  'external_scope':'fresh128 content-balanced TextVQA:64numeric64text, not overall official-distribution estimate; no method tuning on external outcomes',
  'jobs':jobs,'producer_config_sha256':producer_configs,'external_codec_inputs_sha256':codec,
  'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [dev_path,read_path,ROOT/'configs/a2/docvqa_next_confirm256.json',
   ROOT/'configs/data/docvqa_next_references.jsonl',ROOT/'configs/a2/textvqa_balanced128.json',ROOT/'configs/data/textvqa_balanced128_references.json']},
  'source_sha256':{p.name:sha(p) for p in (ROOT/'code/a2').glob('*.py')},
  'inference_source_sha256':{p.name:sha(p) for p in (ROOT/'code/a2').glob('*.py') if p.name.endswith('bank.py') or p.name.startswith(('fused_','session_')) or p.name in ['session.py','quantizers.py','factory.py','run_variants.py','anchor_protocol_v2.py']},
  'declaration':'frozen after complete independent-source development, before any confirm256 or balanced128 inference; no future outcomes or reference branch online'}
 save(frozen_path,result);print(json.dumps({'primary':chosen['variant'],'reference':reference['variant'],'storage_dev_target':bool(qualified),
  'read_primary':rchosen['variant'],'read_dev_target':bool(rqualified),'variants':len(variants),'expected_doc_sessions':result['expected_doc_sessions'],'expected_text_sessions':result['expected_text_sessions'],'jobs':jobs}))
if __name__=='__main__':main()
