"""CPU same-q/text local mechanisms; never a task score or fitting input.
Four dense arms use FP64 arithmetic on actual captured FP16 tensors. The
packed remainder compares the captured real reader output, including original
query-batch arithmetic, with the dense CC reference. No universal rho model.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json,math,sys,time
sys.dont_write_bytecode=True
import torch
from round9_calibration_v1 import decode_key_rotated,decoded_value
LAYERS=[0,6,13,20,27]
SAMPLE_SCHEMA='trajkv.round9.mechanism_sample.v1'
DECODE_HELPER_SHA='757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f'
def require(v,m):
 if not v:raise ValueError('ROUND9_MECHANISM: '+m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def squared(x):return float(x.square().sum())
def repeat(x):return x.repeat_interleave(7,1)
def wht_reference(x):
 y=x.double();shape=y.shape
 for width in (1,2,4,8,16,32,64):
  z=y.reshape(*shape[:-1],128//(2*width),2,width);a,b=z[...,0,:],z[...,1,:]
  y=torch.cat((a+b,a-b),-1).reshape(shape)
 return y*(128**-.5)
def validate_sample(s):
 require(s.get('schema')==SAMPLE_SCHEMA and s.get('purpose')=='mechanism_only','calibration or task data cannot masquerade as mechanisms')
 require(s['layer_idx'] in LAYERS and s['q'].shape==(1,28,1,128),'fixed five M1 layers/last single query')
 require(s['q_rotated'].shape==s['q'].shape and s['packed_output_heads'].shape==s['q'].shape,'query/output shape differs')
 require(s['book'].shape==(16,4),'original low G4_16 only')
 require(s['visual_k'].shape==s['visual_v'].shape and s['visual_k'].shape[:2]==(1,4),'native KV geometry')
 require(s['exact_keys'].shape==s['exact_values'].shape and s['exact_keys'].shape[:2]==(1,4),'same student text geometry')
 require(s['visual_positions'].numel()==s['visual_k'].shape[-2] and s['exact_positions'].numel()==s['exact_keys'].shape[-2],'positions/token support differs')
 require(s['query_positions'].numel()==1 and int(s['visual_positions'].max())<int(s['query_positions'][0]),'post-boundary visual support required')
 require(math.isclose(float(s['scaling']),128**-.5,rel_tol=0,abs_tol=1e-12),'native scale differs')
 for key in ('q','q_rotated','visual_k','visual_v','exact_keys','exact_values','packed_output_heads'):
  require(s[key].device.type=='cpu' and s[key].dtype==torch.float16 and bool(torch.isfinite(s[key]).all()),'captured finite CPU FP16 required '+key)
 return s

def analyze_sample(s,wo,*,fixture_only=False):
 validate_sample(s);require(wo.ndim==2 and wo.shape[1]==3584 and (fixture_only or wo.shape[0]==3584),'exact private W_O geometry')
 require(wo.device.type=='cpu' and bool(torch.isfinite(wo).all()),'CPU finite W_O only')
 q=s['q'].double();qr=s['q_rotated'].double();fields=s['r7_fields'];kr=decode_key_rotated(fields,s['book']).double();vc=decoded_value(fields).double()
 vf=s['visual_v'].double();tk=s['exact_keys'].double();tv=s['exact_values'].double();scale=float(s['scaling'])
 native_vis=q@repeat(s['visual_k']).double().transpose(-1,-2)*scale
 quant_vis=qr@repeat(kr).transpose(-1,-2)*scale+(q*repeat(fields['direction_mean']).double()).sum(-1,keepdim=True)*scale
 text=q@repeat(tk).transpose(-1,-2)*scale
 positions=torch.cat((s['visual_positions'],s['exact_positions']));mask=positions[None,:]<=s['query_positions'][:,None]
 def distributions(vis):
  logits=torch.cat((vis,text),-1).masked_fill(~mask[None,None],-torch.inf);logp=logits.log_softmax(-1)
  return logp.exp(),logp
 pf,lf=distributions(native_vis);pc,lc=distributions(quant_vis);n=vf.shape[-2]
 def output(p,visual):return p[...,:n]@repeat(visual)+p[...,n:]@repeat(tv)
 arms={'FF':output(pf,vf),'CF':output(pc,vf),'FC':output(pf,vc),'CC':output(pc,vc)}
 def proj(heads):return torch.nn.functional.linear(heads.transpose(1,2).reshape(1,1,3584),wo.double())
 z={key:proj(value) for key,value in arms.items()}
 ek=z['CF']-z['FF'];ev_clean=z['FC']-z['FF'];ev_dirty=z['CC']-z['CF'];interaction=z['CC']-z['CF']-z['FC']+z['FF'];total=z['CC']-z['FF']
 phi_k=.5*((z['CF']-z['FF'])+(z['CC']-z['FC']));phi_v=.5*((z['FC']-z['FF'])+(z['CC']-z['CF']))
 packed=proj(s['packed_output_heads'].double());remainder=packed-z['CC']
 losses={key:squared(value-z['FF']) for key,value in z.items()}
 shapley_loss_k=.5*(losses['CF']+losses['CC']-losses['FC']);shapley_loss_v=.5*(losses['FC']+losses['CC']-losses['CF'])
 dot=float((ek*ev_dirty).sum());den=math.sqrt(squared(ek)*squared(ev_dirty));cos=dot/den if den>0 else None
 closure=max(float((total-ek-ev_dirty).abs().max()),float((total-phi_k-phi_v).abs().max()))
 require(closure<=1e-9*(1+float(total.abs().max())),'local vector decomposition identity failed')
 delta=quant_vis-native_vis;beta=delta.mean(-1,keepdim=True);eta=delta-beta
 logvf=native_vis.log_softmax(-1);logvc=quant_vis.log_softmax(-1)
 kl_visual=(logvf.exp()*(logvf-logvc)).sum(-1)
 lmf=torch.logsumexp(lf[...,:n],-1);lmc=torch.logsumexp(lc[...,:n],-1)
 ltf=torch.logsumexp(lf[...,n:],-1);ltc=torch.logsumexp(lc[...,n:],-1)
 mf,mc=lmf.exp(),lmc.exp();tf=ltf.exp()
 mass_kl=mf*(lmf-lmc)+tf*(ltf-ltc)
 text_visible=mask[0,n:];kl_text=(lf[...,n:][...,text_visible].exp()*((lf[...,n:]-ltf[...,None]-lc[...,n:]+ltc[...,None])[...,text_visible])).sum(-1)
 joint_kl=torch.where(mask[None,None],pf*(lf-lc),torch.zeros_like(pf)).sum(-1)
 chain=mass_kl+mf*kl_visual+kl_text
 require(bool(torch.isfinite(chain).all()) and float((joint_kl-chain).abs().max())<1e-8,'probability KL chain identity failed')
 logmoment=torch.logsumexp(quant_vis,-1)-torch.logsumexp(native_vis,-1)
 moment_identity=beta.squeeze(-1)+torch.logsumexp(logvf+eta,-1)
 require(float((logmoment-moment_identity).abs().max())<1e-8,'exact log-partition identity failed')
 dv=repeat(vc-vf);a=pc[...,:n];weighted=a@dv
 diag=(a.square()*dv.square().sum(-1)[:,:,None,:]).sum(-1);weighted_total=weighted.square().sum(-1);cross=weighted_total-diag
 native_bridge=(wht_reference(kr)+fields['direction_mean'].double())
 k_mse_native=float((native_bridge-s['visual_k'].double()).square().mean());k_mse_rotated=float((kr-s['visual_k_rotated'].double()).square().mean()) if 'visual_k_rotated'in s else None
 v_error=vc-vf
 result={'source_id':s['source_id'],'id':s['id'],'dataset':s['dataset'],'layer_idx':s['layer_idx'],'phase':s['phase'],'step':s['step'],
  'visual_tokens':n,'scope':'same captured student q/text; dense FP64 local operator, not task/free-generation quality',
  'arm_definition':'first axis K, second V; F=native visual source, C=packed reconstruction; text fixed',
  'projected_error_SSE':losses,'routing_K_component_SSE':squared(ek),'V_clean_component_SSE':squared(ev_clean),
  'V_dirty_component_SSE':squared(ev_dirty),'interaction_vector_SSE':squared(interaction),
  'vector_Sh_K_SSE':squared(phi_k),'vector_Sh_V_SSE':squared(phi_v),
  'projected_loss_Sh_K':shapley_loss_k,'projected_loss_Sh_V':shapley_loss_v,
  'K_Vdirty_inner_product':dot,'K_Vdirty_cosine':cos,'energy_identity_residual':losses['CC']-squared(ek)-squared(ev_dirty)-2*dot,
  'vector_identity_max_abs':closure,'loss_Sh_sum_residual':losses['CC']-shapley_loss_k-shapley_loss_v,
  'captured_packed_vs_dense_CC_SSE':squared(remainder),'captured_packed_total_vs_FF_SSE':squared(packed-z['FF']),
  'packed_input_query_tokens':s.get('packed_input_query_tokens'),'packed_remainder_includes_actual_batch_rounding':True,
  'C1':{'K_native_bridge_MSE':k_mse_native,'K_actual_rotated_MSE':k_mse_rotated,'V_actual_sequential_half_MSE':float(v_error.square().mean()),
     'weighted_V_error_scope':'per query-head BEFORE W_O, full-joint visual a; cross is deterministic token off-diagonal innerproducts, not a stochastic rho',
     'V_weighted_total_by_queryhead':weighted_total.reshape(-1).tolist(),'V_weighted_diag_by_queryhead':diag.reshape(-1).tolist(),'V_weighted_cross_by_queryhead':cross.reshape(-1).tolist(),
     'V_weighted_total_mean':float(weighted_total.mean()),'V_weighted_diag_mean':float(diag.mean()),'V_weighted_cross_mean':float(cross.mean()),
     'K_native_mean_error_norm':float((native_bridge-s['visual_k'].double()).mean(-2).norm(dim=-1).mean()),'V_mean_error_norm':float(v_error.mean(-2).norm(dim=-1).mean())},
  'routing':{'visual_mass_F_mean':float(mf.mean()),'visual_mass_C_mean':float(mc.mean()),'modality_mass_KL_mean':float(mass_kl.mean()),
    'within_visual_KL_mean':float(kl_visual.mean()),'within_visual_TV_mean':float(.5*(logvf.exp()-logvc.exp()).abs().sum(-1).mean()),
    'within_visual_top1_changed_fraction':float((native_vis.argmax(-1)!=quant_vis.argmax(-1)).double().mean()),
    'joint_attention_KL_mean':float(joint_kl.mean()),'KL_chain_max_abs':float((joint_kl-chain).abs().max()),
    'score_common_beta_RMS':float(beta.square().mean().sqrt()),'score_centered_eta_RMS':float(eta.square().mean().sqrt()),
    'exact_visual_log_partition_delta_mean':float(logmoment.mean()),'logmoment_identity_max_abs':float((logmoment-moment_identity).abs().max())},
  'no_universal_variance_fraction_or_causal_task_attribution':True,'training_eligible':False,'quality_scored':False,'fixture_only':fixture_only}
 return result


def analyze_collection(directory,output):
 directory,output=Path(directory),Path(output)
 source_before=sha(__file__);helper=Path(__file__).with_name('round9_calibration_v1.py')
 require(sha(helper)==DECODE_HELPER_SHA,'frozen physical decoder helper changed')
 if not(directory/'complete.json').is_file():return {'status':'PENDING_REAL_MECHANISM_COLLECTION','analysis_written':False}
 complete=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
 require(complete['status']=='PASS_MECHANISM_32_REAL_MODEL_SOURCES' and complete['purpose']=='mechanism_only' and complete['actual_GPU'] is True,'no actual complete32 mechanism collection')
 require(complete['capture_layers']==LAYERS and len(complete['sources'])==complete['expected_sources']==32,'fixed capture source/layer count')
 require(sha(directory/'manifest.json')==complete['manifest_sha256'],'collection manifest SHA differs')
 require(sha(complete['wo_file'])==complete['wo_sha256'],'private W_O SHA differs')
 wos=torch.load(complete['wo_file'],map_location='cpu',weights_only=True);require(set(wos)==set(LAYERS),'only five private W_O layers expected')
 output.mkdir(parents=True,exist_ok=False);results=[];sourceproof=[];started=time.time()
 for source in complete['sources']:
  path=Path(source['file']);require(sha(path)==source['sha256'],'private mechanism shard SHA differs')
  case=torch.load(path,map_location='cpu',weights_only=True)
  require(case['schema']=='trajkv.round9.mechanism_case.v1' and case['purpose']=='mechanism_only' and case['training_eligible'] is False,'wrong case role')
  require(set(case['layers'])==set(LAYERS) and {e['layer_idx'] for e in case['events']}==set(LAYERS),'captured layer support differs')
  for event in case['events']:
   i=event['layer_idx'];sample={**case['layers'][i],**event,'book':case['book'],'visual_positions':case['visual_positions']}
   results.append(analyze_sample(sample,wos[i]))
  sourceproof.append(source)
 require(len({r['source_id'] for r in results})==32 and len({r['source_id'] for r in results if r['dataset']=='infovqa'})==16,'Info source count differs')
 require(len({r['source_id'] for r in results if r['dataset']=='textvqa'})==16,'Text source count differs')
 with (output/'records.jsonl').open('x',encoding='utf-8') as f:
  for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
 groups=defaultdict(list)
 for r in results:groups[(r['dataset'],r['layer_idx'],r['phase'])].append(r)
 aggregate=[]
 for (ds,layer,phase),rows in sorted(groups.items()):
  by_source=defaultdict(list)
  for row in rows:by_source[row['source_id']].append(row)
  def mean(get):return sum(sum(get(r) for r in rs)/len(rs) for rs in by_source.values())/len(by_source)
  aggregate.append({'dataset':ds,'layer':layer,'phase':phase,'sources':len(by_source),'local_events':len(rows),
   'projected_total_SSE':mean(lambda r:r['projected_error_SSE']['CC']),'projected_Sh_K':mean(lambda r:r['projected_loss_Sh_K']),
   'projected_Sh_V':mean(lambda r:r['projected_loss_Sh_V']),'interaction_SSE':mean(lambda r:r['interaction_vector_SSE']),
   'packed_remainder_SSE':mean(lambda r:r['captured_packed_vs_dense_CC_SSE']),'K_rotated_MSE':mean(lambda r:r['C1']['K_actual_rotated_MSE']),
   'V_MSE':mean(lambda r:r['C1']['V_actual_sequential_half_MSE']),'V_weighted_diag':mean(lambda r:r['C1']['V_weighted_diag_mean']),
   'V_weighted_cross':mean(lambda r:r['C1']['V_weighted_cross_mean'])})
 require(sha(__file__)==source_before and sha(helper)==DECODE_HELPER_SHA,'analysis sources changed during execution')
 summary={'schema':'trajkv.round9.local_mechanism_analysis.v1','status':'PASS_SAME_STATE_LOCAL_OPERATOR_ANALYSIS_ONLY','QA_sources':32,'captured_layer_count':5,
  'actual_events':len(results),'dense_arm_evaluations':4*len(results),'new_model_forward_or_generation':0,'training_eligible':False,'task_quality_eligible':False,
  'aggregation':'source mean within dataset/layer/phase; no task scores or universal rho shares','rows':aggregate,
  'collection_complete_sha256':sha(directory/'complete.json'),'collection_manifest_sha256':sha(directory/'manifest.json'),
  'private_WO_sha256':complete['wo_sha256'],'private_shards':sourceproof,'records_sha256':sha(output/'records.jsonl'),
  'analyzer_source_sha256':sha(__file__),'decode_helper_source_sha256':sha(Path(__file__).with_name('round9_calibration_v1.py')),
  'source_model_weights_or_private_tensors_must_not_be_published':True,'wall_s':time.time()-started}
 write(output/'summary.json',summary)
 if aggregate:
  with(output/'table.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
 write(output/'complete.json',{'status':summary['status'],'summary_sha256':sha(output/'summary.json'),'records_sha256':summary['records_sha256'],'task_quality_eligible':False})
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 require(1<=a.threads<=8,'bounded CPU thread count1..8');torch.set_num_threads(a.threads)
 r=analyze_collection(a.collection,a.output);print(json.dumps({k:r[k] for k in ('status','QA_sources','actual_events') if k in r}),flush=True)
if __name__=='__main__':main()
