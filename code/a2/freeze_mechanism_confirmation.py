"""Freeze the mechanism-led new64 package from complete development evidence.

This does not launch inference. New64 outcomes must not exist at freezing.
Materialized ports and oracle tensors cannot be storage candidates.
"""
import argparse,copy,hashlib,json,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/analysis'))
from remote import REMOTE_ROOT,PYTHON
from analyze_external import score_docvqa
from costs import visual_owner,role

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps({k:x for k,x in v.items() if k!='name'},sort_keys=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--extra-runs',nargs='*',default=[]);a=p.parse_args()
 frozen=ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json'
 if frozen.exists():raise RuntimeError('Freeze exists; do not overwrite')
 if any((ROOT/'results'/f'mechanism_confirm64_s{s}_v1').exists() for s in (0,1)):raise RuntimeError('Confirmation outcomes already exist')
 names=['mad_reference100_gpu1_v1','mad100_e2e_s0_v1','mad100_e2e_s1_v1','final100_e2e_s0_v1','final100_e2e_s1_v1']+a.extra_runs
 ordered_refs=json.loads((ROOT/'configs/a2/mechanism100_refs.json').read_text());refs={r['id']:r for r in ordered_refs}
 calibration_ids={r['id'] for r in ordered_refs[:20]};selection_ids=set(refs)-calibration_ids;groups=defaultdict(list);inputs={}
 for name in names:
  folder=ROOT/'results'/name
  if json.loads((folder/'complete.json').read_text()).get('status')!='COMPLETE':raise RuntimeError('Incomplete development run: '+name)
  for f in folder.rglob('records.jsonl'):
   inputs[str(f.relative_to(ROOT))]=sha(f)
   for line in f.read_text().splitlines():
    r=json.loads(line)
    if r['session_id'] not in refs or r.get('protocol')!='image_boundary_free_running':continue
    if r.get('task_accuracy_eligible') is False or r.get('primary_quality_eligible') is False:continue
    groups[r['variant']].append(r)
 table=[]
 for name,rows in groups.items():
  if len(rows)!=100 or {r['session_id'] for r in rows}!=set(refs):raise RuntimeError('100 matched development images required: '+name)
  cfg=rows[0]['variant_config']
  if not all(canonical(r['variant_config'])==canonical(cfg) for r in rows):raise RuntimeError('Config drift')
  scored={r['session_id']:score_docvqa(r['turns'][0]['answer'],refs[r['session_id']]['answer'])['score'] for r in rows}
  score=sum(scored.values())/100;selection_score=sum(scored[k] for k in selection_ids)/len(selection_ids)
  total=sum(visual_owner(r)[0] for r in rows);den=sum(r['owned_initial']['visual_fp16_denominator'] for r in rows)
  encoder=sum(r.get('encoder_calibration_parameters_bytes',0) for r in rows)
  table.append({'name':name,'images':100,'anls':score,'selection_images':80,'selection_anls':selection_score,
    'effective_bits':16*total/den,'budget_effective_bits_including_encoder_constants':16*(total+encoder)/den,
    'encoder_constants_bytes':encoder/100,'mean_visual_bytes':total/100,'role':role(rows[0]),'config':cfg})
 by={r['name']:r for r in table}
 reference_names=['fp16_reference100','base2_reference','native4_reference','kvarn2_reference']
 for n in ('calib2_packed_correct','calib2_fp32_control','post_base2'):
  if n in by:reference_names.append(n)
 quality_ref=sorted([by[n] for n in reference_names],key=lambda r:(-r['selection_anls'],reference_names.index(r['name'])))[0]
 eligible=[r for r in table if r['role']=='actual packed runtime' and r['budget_effective_bits_including_encoder_constants']<=1.8]
 qualified=[r for r in eligible if r['selection_anls']>=quality_ref['selection_anls']-.005]
 if qualified:primary=sorted(qualified,key=lambda r:(r['budget_effective_bits_including_encoder_constants'],-r['selection_anls'],r['name']))[0]
 else:primary=sorted(eligible,key=lambda r:(-r['selection_anls'],r['budget_effective_bits_including_encoder_constants'],r['name']))[0]
 chosen=[];seen=set();aliases={}
 def add(label,cfg):
  v=copy.deepcopy(cfg);v['name']=label;key=canonical(v)
  if key in seen:
   aliases[label]=next(x['name'] for x in chosen if canonical(x)==key);return
  seen.add(key);chosen.append(v)
 add('storage_primary',primary['config'])
 labelled=[('full8','fp16_reference100'),('base2_8','base2_reference'),('native4_8','native4_reference'),('kvarn2_8_quality_port','kvarn2_reference'),
    ('native_k2v1_8','native_k2v1'),('k2_lloyd_v1','k2_vlloyd_channel'),('k2_global_mad_v1','k2_vmad_global'),
    ('lloyd_joint','lloyd_joint_none'),('lloyd_rms125_joint','lloyd_fold_rms_1.25'),('rtn_rms125_joint','rtn_fold_rms_1.25'),
    ('direction_vq2_v1','direction_vq4_norm_v1_lloyd'),('direction_k1_v2','direction_sign_norm_v2_channel'),
    ('calibration_warm','coupled_warm_k2v1'),('calibration_none_rms','k2_rms_vlloyd_uncalibrated'),
    ('calib2_fp32','calib2_fp32_control'),('calib_k2v1_fp32','calib_k2v1_fp32_control'),
    ('calib1_mean_fp32','calib1_center_fp32_100'),('calib1_mean_fp16','calib1_center_fp16_100')]
 for label,name in labelled:add(label,by[name]['config'])
 add('kvarn2_k_clipped_v1',{'method':'hybrid_packed','key_codec':'kvarn2','clip_v':.3,'cap':8,'selector':'none','store_selector_meta':False})
 reference_labels={source:label for label,source in labelled}
 quality_reference_label=reference_labels.get(quality_ref['name'])
 if quality_reference_label is None:
  quality_reference_label='quality_reference_extra';add(quality_reference_label,quality_ref['config'])
 # Confirm the stronger directional effect using Lloyd's actual quantized norm.
 for arm,mode in [('A00','quant_norm_quant_direction'),('A10','true_norm_quant_direction'),('A01','quant_norm_true_direction')]:
  add('oracle_lloyd_'+arm,{'method':'oraclek','encoder':'lloyd','oracle_mode':mode,'cap':8,'selector':'none','store_selector_meta':False})
 for policy in ('none','all','periodic','prefill_only','random','mass','risk','thumb_random'):
  cfg={'method':'asym','selector':policy,'page':128,'cap':16,'store_selector_meta':True,'selector_query_last':True}
  if policy!='all':cfg.update(fraction=.25,global_read_equivalents=32)
  add('read_'+policy,cfg)
 add('full16',{'method':'full','selector':'none','cap':16,'store_selector_meta':False})
 add('base2_16',{'method':'base2','selector':'none','cap':16,'store_selector_meta':False})
 dev=json.loads((ROOT/'configs/a2/mechanism100.json').read_text(encoding='utf-8'));confirm=json.loads((ROOT/'configs/a2/confirm64_short.json').read_text(encoding='utf-8'))
 assert len(confirm)==64 and len({r['image_id'] for r in confirm})==64 and not {r['image_id'] for r in confirm}&{r['image_id'] for r in dev}
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());base['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/confirm64_short.json',turns=6);base['variants']=chosen
 jobs=[]
 for shard in (0,1):
  name=f'mechanism_confirm64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=shard*32,count=32)
  (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  # Dependencies are the complete current exploration queues, preserving one
  # model worker per GPU and keeping source snapshots immutable.
  dep=f'direction100_s{shard}_v1'
  queue={'wait_complete':REMOTE_ROOT+f'/results/{dep}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{dep}.launch.json',
   'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(queue,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 codec_inputs={}
 for v in chosen:
  for key in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path'):
   if not v.get(key):continue
   remote=v[key]
   if not remote.startswith(REMOTE_ROOT+'/'):raise RuntimeError('External codec input outside audited project')
   local=ROOT/remote[len(REMOTE_ROOT)+1:]
   if not local.is_file():raise RuntimeError('Missing frozen codec input: '+str(local))
   codec_inputs[remote]=sha(local)
 result={'schema':'trajkv.mechanism.confirmation.v2','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
  'primary_candidate':primary,'quality_reference':quality_ref,'quality_reference_confirmation_variant':quality_reference_label,'development_quality_target_met':bool(qualified),'target_margin':.005,
  'selection_rule':'on80 images excluding the20 calibration images: lowest conservative bits among <=1.8 actual packed candidates within0.5pp of the prespecified strong-reference winner; otherwise highest validation quality<=1.8; ties lower bits then name',
  'selection_ids':sorted(selection_ids),'calibration_ids':sorted(calibration_ids),
  'quality_reference_rule':'highest80-image mean among prespecified Full/Base2/Native4/KVarN quality-port and corrected Calib2/FP32 Base2 controls, same cap8; fixed before new64',
  'primary_tests':[['storage_primary','k2_lloyd_v1'],['storage_primary',quality_reference_label],['storage_primary','calibration_warm'],['read_periodic','read_all'],['read_prefill_only','read_all'],['oracle_lloyd_A01','oracle_lloyd_A10']],
  'statistics':'question-weighted ANLS/exact; paired bootstrap resamples whole images; report point 0.5pp target separately from interval-based noninferiority; primary superiority tests Holm-adjusted',
  'read_target':'same stored representation as read_all; >=50% reduction of residual+selection logical bytes with quality delta >=-0.005; report total-cache traffic and timing independently',
  'read_budget_note':'read_all fully refines every read; other read policies have the same global cap .25*32 full residual forward equivalents including selection fees',
  'dev_table':table,'dev_inputs':inputs,'external_codec_inputs_sha256':codec_inputs,'confirm_membership_sha256':sha(ROOT/'configs/a2/confirm64_short.json'),
  'confirm_ids':[r['id'] for r in confirm],'confirm_images':64,'confirm_questions':sum(min(6,len(r['questions'])) for r in confirm),
  'variants':chosen,'aliases':aliases,'jobs':jobs,'expected_sessions':64*len(chosen),
  'source_sha256':{p.name:sha(p) for p in (ROOT/'code/a2').glob('*.py')},
  'declaration':'before new64 inference; confirmation outcomes unavailable; all exploratory parameter decisions remain confined to old image pool'}
 frozen.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('development_quality_target_met','confirm_images','confirm_questions','expected_sessions')}|{'primary':primary['name'],'reference':quality_ref['name'],'variants':len(chosen)}))
if __name__=='__main__':main()
