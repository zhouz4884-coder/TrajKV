"""Prospective candidate selection from dev8 only, then immutable new64 configs."""
import copy,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/analysis'))
from remote import REMOTE_ROOT,PYTHON
from analyze_external import score_docvqa

def visual_bytes(r):
 return sum(r['owned_initial'].get(k,0) for k in ('visual_base','visual_residual','selector_metadata','hadamard_constant','visual_position'))
def main():
 out=ROOT/'audit/a2/frozen_confirmation.json'
 if out.exists():raise RuntimeError('Confirmation already frozen; do not overwrite')
 runs=['a2_short_dev4_s0_v1','a2_short_dev4_s1_v1','a2_vfloor_dev_gpu0_v1','a2_vfloor8_dev_gpu1_v1','a2_final_repr_dev_gpu0_v1']
 records=[]
 for run in runs:
  folder=ROOT/'results'/run
  receipt=json.loads((folder/'complete.json').read_text())
  if receipt.get('status')!='COMPLETE':raise RuntimeError('Development dependency incomplete')
  record_path=folder/'quality/records.jsonl'
  if not record_path.exists():record_path=folder/'records.jsonl'
  records += [json.loads(s) for s in record_path.read_text().splitlines()]
 dev=json.loads((ROOT/'configs/a2/dev8_short.json').read_text(encoding='utf-8'));dev_ids={r['id'] for r in dev}
 refs={r['id']:r['references'] for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text(encoding='utf-8').splitlines()) if r['id'] in dev_ids}
 groups=defaultdict(list)
 for r in records:
  assert r['session_id'] in dev_ids and r['protocol']=='image_boundary_free_running'
  groups[r['variant']].append(r)
 table=[]
 for name,rs in groups.items():
  assert len(rs)==8 and {r['session_id'] for r in rs}==dev_ids
  scores=[score_docvqa(t['answer'],refs[r['session_id']][t['turn']-1]['answer'])['score'] for r in rs for t in r['turns']]
  bits=16*sum(visual_bytes(r) for r in rs)/sum(r['owned_initial']['visual_fp16_denominator'] for r in rs)
  table.append({'name':name,'anls':sum(scores)/len(scores),'questions':len(scores),'effective_visual_bits':bits,
    'mean_visual_bytes':sum(visual_bytes(r) for r in rs)/8,'config':rs[0]['variant_config']})
 novel_names={'hybrid_packed_8','hybrid_packed_16','native2_clipped1_8','native2_clipped1_16','vlate1','vlate2','vearly2','vmid2','vlate2_kclip08','vlate2_kclip06','kclip08_only'}
 eligible=[r for r in table if (r['name'] in novel_names or (r['config']['method']=='rate' and
             (r['config'].get('v_layers') or r['config'].get('k_clip',1.)!=1. or r['config'].get('layer_bits_k') or r['config'].get('layer_bits_v') or
              r['config'].get('k_encoder','rtn')!='rtn' or r['config'].get('v_encoder','rtn')!='rtn'))) and r['effective_visual_bits']<=1.8]
 if not eligible:raise RuntimeError('No eligible low-storage prototype')
 primary=sorted(eligible,key=lambda r:(-r['anls'],r['mean_visual_bytes'],r['name']))[0]
 cap=primary['config'].get('cap',16)
 names=['full8','full16','base2','native4_8','kvarn2_8','kvarn2_16','base1','k_only','v_only',
   'native_k2v1_8','native_k2v1_16','a2_none','a2_random','a2_mass','a2_risk','a2_layer_uniform','a2_thumb_random','a2_all',
   ]
 chosen={r['name']:copy.deepcopy(r['config']) for r in table if r['name'] in names}
 chosen[primary['name']]=copy.deepcopy(primary['config'])
 chosen['a2_periodic']={'name':'a2_periodic','method':'asym','selector':'periodic','fraction':.25,'global_read_equivalents':32,'page':128,'cap':16,'selector_query_last':True}
 for row in table:
  cfg=row['config']
  if cfg.get('v_encoder') in ('lloyd','weighted_lloyd') and cfg.get('bits_k',1)==2:chosen[row['name']]=copy.deepcopy(cfg)
 calib=sorted([r for r in table if r['name'] in ('calib1_post','calib_k2v1_tau0')],key=lambda r:(-r['anls'],r['mean_visual_bytes'],r['name']))[0]
 chosen[calib['name']]=copy.deepcopy(calib['config'])
 # Keep the other two executable codec combinations as prespecified secondary
 # comparisons; caps chosen on development quality, with bytes breaking ties.
 for family in [('hybrid_packed_8','hybrid_packed_16'),('native2_clipped1_8','native2_clipped1_16')]:
  option=sorted([r for r in table if r['name'] in family],key=lambda r:(-r['anls'],r['mean_visual_bytes'],r['name']))[0]
  chosen[option['name']]=copy.deepcopy(option['config'])
 if primary['config']['method']=='rate' and primary['config'].get('v_layers'):
  n=len(primary['config']['v_layers'])
  for label,layers in [('matched_early',list(range(n))),('matched_middle',list(range(14-n//2,14-n//2+n))),('matched_late',list(range(28-n,28)))]:
   cfg=copy.deepcopy(primary['config']);cfg.update(name=label,k_clip=1.,v_layers=layers)
   if layers==primary['config']['v_layers'] and primary['config'].get('k_clip',1.)==1.:continue
   chosen[label]=cfg
 # Exactly the same image-resolution reference as the storage candidate.
 chosen['native4_samecap']={'name':'native4_samecap','method':'native4','cap':cap,'selector':'none','store_selector_meta':False}
 chosen['base2_samecap']={'name':'base2_samecap','method':'base2','cap':cap,'selector':'none','store_selector_meta':False}
 # These are conditional-quality diagnostics, with actual materialized owners
 # reported separately; they never enter the storage-candidate selection.
 for bits in (1,2):
  name=f'fp16k_v{bits}_diagnostic';chosen[name]={'name':name,'method':'axisdiag','axis':'v','axis_bits':bits,'layout':'token','cap':cap,'selector':'none'}
 # Avoid repeating identical fixed-codec/cap references under two names.
 seen_fixed={};remove=[]
 for name,cfg in chosen.items():
  if cfg['method'] in ('full','base1','base2','native4','kvarn2'):
   key=(cfg['method'],cfg.get('cap',16))
   if key in seen_fixed:remove.append(name)
   else:seen_fixed[key]=name
 for name in remove:chosen.pop(name)
 conf=json.loads((ROOT/'configs/a2/confirm64_short.json').read_text(encoding='utf-8'))
 ids={r['id'] for r in conf};assert len(conf)==64 and len(ids)==64 and not ids&dev_ids
 base=json.loads((ROOT/'configs/a2/a2_short_dev4_s0_v1.json').read_text());base['arguments']['samples']=REMOTE_ROOT+'/configs/a2/confirm64_short.json';base['variants']=list(chosen.values())
 jobs=[]
 for shard in (0,1):
  cfg=copy.deepcopy(base);cfg['start']=32*shard;cfg['count']=32;name=f'a2_confirm64_s{shard}_v1'
  local=ROOT/'configs/a2'/(name+'.json');local.write_text(json.dumps(cfg,indent=2))
  j={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
    'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_variants.py','--config',REMOTE_ROOT+'/configs/a2/'+name+'.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/(name+'.json')).write_text(json.dumps(j,indent=2));jobs.append(name)
 from datetime import datetime,timezone
 frozen={'schema':'trajkv.a2.confirmation.v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'primary_candidate':primary,'dev_table':table,'selection_rule':'highest dev ANLS among executable novel prototypes <=1.8 actual effective visual bits; ties lower bytes then name',
  'primary_contrasts':[[primary['name'],'native_k2v1_'+str(cap)],['a2_risk','a2_random'],['a2_risk','a2_mass']],
  'quality_reference':[primary['name'],'full'+str(cap)],'quality_target_margin':.005,
  'confirm_membership_sha256':hashlib.sha256((ROOT/'configs/a2/confirm64_short.json').read_bytes()).hexdigest(),
  'confirm_images':64,'confirm_questions':sum(min(6,len(r['questions'])) for r in conf),'confirm_ids':[r['id'] for r in conf],
  'variants':base['variants'],'jobs':jobs,'expected_records':64*len(chosen),
  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code/a2').glob('*.py')},
  'declaration':'written before confirmation inference; no confirmation outcomes used; bare-prompt development retained separately'}
 out.write_text(json.dumps(frozen,indent=2));print(json.dumps({'primary':primary['name'],'dev_anls':primary['anls'],'bits':primary['effective_visual_bits'],'variants':len(chosen),'confirm_questions':frozen['confirm_questions'],'records':frozen['expected_records']}))
if __name__=='__main__':main()
