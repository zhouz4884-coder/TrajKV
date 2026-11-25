"""Real M1 native-prefix diagnostic map. Zero answers; no production selector query leakage."""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2); f.write('\n')

def wht(x):
    y=x.float(); shape=y.shape
    for width in (1,2,4,8,16,32,64):
        z=y.reshape(*shape[:-1],128//(2*width),2,width); a,b=z[...,0,:],z[...,1,:]
        y=__import__('torch').cat((a+b,a-b),-1).reshape(shape)
    return y*(128**-.5)

def spectrum(x):
    import torch
    x=x.float(); n=x.shape[0]; mu=x.mean(0,keepdim=True); c=x-mu
    eig=torch.linalg.eigvalsh(c.T@c/max(n,1)).clamp_min(0).flip(0)
    den=eig.sum().clamp_min(1e-30)
    return {'mean_energy_fraction':float(mu.square().sum()*n/x.square().sum().clamp_min(1e-30)),
            'centered_top32_fraction':float(eig[:32].sum()/den),
            'centered_top64_fraction':float(eig[:64].sum()/den)}

def spatial(x,seed):
    import torch
    n=x.shape[0]; require(n%256==0,'tile geometry')
    grid=x.float().reshape(-1,16,16,128)
    def residual(z):
        blocks=z.reshape(-1,8,2,8,2,128)
        return float((blocks-blocks.mean((2,4),keepdim=True)).square().sum()/z.square().sum().clamp_min(1e-30))
    gen=torch.Generator(device='cpu').manual_seed(seed)
    sh=torch.randperm(256,generator=gen).to(x.device)
    shuffled=grid.reshape(-1,256,128).index_select(1,sh).reshape_as(grid)
    return {'two_by_two_residual_energy_fraction':residual(grid),
            'within_tile_shuffled_residual_energy_fraction':residual(shuffled)}

def require(value,message):
    if not value: raise ValueError('ROUND13_STAGE0: '+message)

def local_map(q,k,v,quant_scores,cv,tk,tv,rank,random_rank,packed_cc):
    import torch, math
    q=q.float(); rep=lambda t:t.float().repeat_interleave(7,1)
    lf=q@rep(k).transpose(-1,-2)/math.sqrt(128)
    lc=quant_scores.float()
    lt=q@rep(tk).transpose(-1,-2)/math.sqrt(128)
    n=k.shape[-2]
    pf=torch.cat((lf,lt),-1).softmax(-1); pc=torch.cat((lc,lt),-1).softmax(-1)
    def out(p,values): return p[...,:n]@rep(values)+p[...,n:]@rep(tv)
    of=out(pf,v); oc=out(pc,cv); ref=float((oc-of).square().sum())
    mf=pf[...,:n].sum(-1).clamp_min(1e-30); mc=pc[...,:n].sum(-1).clamp_min(1e-30)
    result={'native_visual_mass_mean':float(mf.mean()),'compressed_visual_mass_mean':float(mc.mean()),
            'CC_vs_FF_output_heads_SSE':ref,
            'actual_packed_CC_vs_dense_CC_max_abs':float((packed_cc.float()-oc).abs().max()),
            'actual_packed_CC_vs_FF_SSE':float((packed_cc.float()-of).square().sum()),
            'scope':'fixed native question query/text; decoded Half fields and actual Half WHT query, dense FP32 local pre-WO operator, not task benefit',
            'curves':[],'interventions':[]}
    for name,order in (('image_only_best',rank),('image_only_random',random_rank)):
        for count in sorted(set(min(n,max(1,int(h))) for h in (32,55,.49*n,.72*n))):
            ids=order[:count].to(q.device)
            result['curves'].append({'selector':name,'rows':count,
                'native_conditional_visual_mass':float((pf[...,:n].index_select(-1,ids).sum(-1)/mf).mean()),
                'compressed_conditional_visual_mass':float((pc[...,:n].index_select(-1,ids).sum(-1)/mc).mean())})
            if count not in {min(n,32),min(n,55)}: continue
            for arm in ('K','V','KV'):
                scores=lc.clone() if 'K' in arm else lc
                if 'K' in arm: scores[...,ids]=lf[...,ids]
                values=cv.clone() if 'V' in arm else cv
                if 'V' in arm: values[...,ids,:]=v[...,ids,:]
                p=torch.cat((scores,lt),-1).softmax(-1)
                sse=float((out(p,values)-of).square().sum())
                result['interventions'].append({'selector':name,'rows':count,'restore':arm,
                    'output_heads_SSE':sse,'SSE_reduction_from_CC':ref-sse})
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True,type=Path)
    p.add_argument('--inputs',required=True,type=Path); p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    require(sha(a.inputs)==a.inputs_sha256,'calibration input SHA')
    rows=[json.loads(t) for t in a.inputs.read_text(encoding='utf-8-sig').splitlines() if t.strip()]
    require(len(rows)==32 and len({r['id'] for r in rows})==32,'exact32 sources')
    require(all(not(set(r)&{'answer','answers','refs','references','scores','labels'}) for r in rows),'no answer fields')
    require(sum(r['dataset']=='infovqa' for r in rows)==16 and sum(r['dataset']=='docvqa' for r in rows)==16,'Info16Doc16')
    output=a.output.resolve(); require(str(output).startswith('/media/') and '/results/round13' in str(output),'data disk')
    output.mkdir(parents=True,exist_ok=False); (output/'records').mkdir()
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from model_registry import get_model_spec
    from evaluate import model_identity,tensor_sha
    from m1_protocol import prepare_m1_inputs,m1_prefix_forward,m1_forward_suffix,clone_m1_prefix_cache
    from attention_bridge import attention_context,language_attention_modules
    from r7_legacy_bridge_v1 import build_r7_owners
    from round9_calibration_v1 import decode_key_rotated,decoded_value,FIELDS
    from round13_selector_v1 import rank_visual_rows
    import round10_runner_v1 as frozen
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),'RTX3090 CUDA required')
    torch.set_grad_enabled(False); torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter(); identity=model_identity(config,get_model_spec('M1'))
    processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(config['model_path'],dtype=torch.float16,
        device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval().requires_grad_(False)
    params=sum(v.numel() for v in model.parameters())
    require(params==7944373760 and all(v.dtype==torch.float16 for v in model.parameters()),'full original M1 weights')
    variant=frozen.frozen_variant(config)
    kwargs={k:config[k] for k in ('legacy_a2','legacy_experiments','legacy_scripts','kivi_source')}
    kwargs['expected_asset_sha256']=config['expected_asset_sha256']
    manifest={'schema':'trajkv.round13.stage0.v1','status':'RUNNING','actual_CUDA':True,
        'model_identity':identity,'model_parameter_count':params,'input_sha256':sha(a.inputs),'config_sha256':sha(a.config),
        'source_sha256':sha(__file__),'expected_sources':32,'task_answer_events':0,'references_loaded':False,
        'production_memory_evidence':False,'native_sources_retained_for_diagnostic_only':True,
        'question_queries_used_only_for_diagnostic_not_production_selection':True,
        'image_only_selector_frozen_before_diagnostic_results':True,'native_prefix_forwards':0,'native_question_TF_forwards':0}
    write(output/'manifest.json',manifest)
    layers=(0,6,13,20,27); _,mods=language_attention_modules(model); completed=[]
    for i,row in enumerate(rows):
        t=time.perf_counter(); case={'id':row['id'],'dataset':row['dataset'],'status':'RUNNING','spectra':[],'local_maps':[]}
        hooks=[]; pre={}; native=batch=prefix=suffix=clone=owners=capq=None
        try:
            require(sha(row['image_path'])==row['original_image_sha256'],'image content SHA')
            with Image.open(row['image_path']) as raw: image=raw.convert('RGB')
            batch=prepare_m1_inputs(processor,image,row['question']+config['question_suffix'],max_patches=8)
            vp=batch.visual_positions
            for layer in layers:
                def capture(_mod,_inp,out,layer=layer):
                    pre[layer]=out.detach().view(1,out.shape[1],4,128).transpose(1,2).index_select(2,vp).contiguous().clone()
                hooks.append(mods[layer].k_proj.register_forward_hook(capture))
            with attention_context(model,capture_queries=True,window_size=16) as image_cap:
                prefix=m1_prefix_forward(model,batch)
            manifest['native_prefix_forwards']+=1
            for h in hooks: h.remove()
            hooks=[]; native=prefix.past_key_values; prefix=None
            rank,rank_meta=rank_visual_rows(native,vp,selector='importance_diversity',seed=0)
            random_rank,rand_meta=rank_visual_rows(native,vp,selector='random',seed=0)
            owners,meta=build_r7_owners(native,vp,variant,**kwargs)
            clone=clone_m1_prefix_cache(native,batch.boundary)
            with attention_context(model,capture_queries=True,window_size=1) as qcap:
                suffix=m1_forward_suffix(model,batch,clone)
            manifest['native_question_TF_forwards']+=1
            top=suffix.logits[0,-1].float().topk(2)
            case.update(visual_tokens=vp.numel(),compression_boundary=batch.boundary,
                input_ids_sha256=tensor_sha(batch.ids),position_ids_sha256=tensor_sha(batch.positions),
                first_native_suffix_top2_ids=top.indices.tolist(),first_native_suffix_top2_logits=top.values.tolist(),
                selector=rank_meta,random_selector=rand_meta,
                image_prefix_query_capture_counts=image_cap['captured_tokens_by_layer'])
            for layer in layers:
                k=native.layers[layer].keys.index_select(2,vp); v=native.layers[layer].values.index_select(2,vp)
                for head in range(4):
                    for kind,x in (('pre_RoPE_K',pre[layer][0,head]),('post_RoPE_K',k[0,head]),('V',v[0,head])):
                        case['spectra'].append({'layer':layer,'head':head,'kind':kind,
                            **spectrum(x),**spatial(x,1125+i)})
                owner=owners[layer]; fields={name:getattr(owner.state,name) for name in FIELDS if hasattr(owner.state,name)}
                kr=decode_key_rotated(fields,owner.shared['book'])
                cv=decoded_value(fields)
                q=qcap['queries_by_layer'][layer][...,-1:,:]
                allk=clone.layers[layer].keys; allv=clone.layers[layer].values
                mask=torch.ones(allk.shape[-2],device=vp.device,dtype=torch.bool); mask[vp]=False
                tk=allk[:,:,mask].contiguous(); tv=allv[:,:,mask].contiguous()
                from fast_wht import wht as real_wht
                qr=real_wht(q,half=True)
                quant_scores=qr.float()@kr.repeat_interleave(7,1).float().transpose(-1,-2)/(128**.5)
                quant_scores+=(q.float()*fields['direction_mean'].repeat_interleave(7,1).float()).sum(-1,keepdim=True)/(128**.5)
                text_positions=torch.arange(allk.shape[-2],device=vp.device)[mask]
                query_positions=torch.tensor([batch.full_length-1],device=vp.device,dtype=torch.long)
                packed_cc=owner.attend(q,exact_keys=tk,exact_values=tv,exact_positions=text_positions,query_positions=query_positions)
                case['local_maps'].append({'layer':layer,**local_map(q,k,v,quant_scores,cv,tk,tv,rank,random_rank,packed_cc)})
            case['status']='PASS_DIAGNOSTIC_ONLY'; case['wall_seconds']=time.perf_counter()-t
        except Exception as exc:
            case.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        finally:
            for h in hooks: h.remove()
            filename=f'{i:03d}_{hashlib.sha256(row["id"].encode()).hexdigest()[:16]}.json'
            write(output/'records'/filename,case); completed.append({'id':row['id'],'file':filename,'status':case['status']})
            print(json.dumps({'completed':i+1,'expected':32,'id':row['id'],'status':case['status'],
                              'wall_seconds':case.get('wall_seconds'),'error':case.get('error')},ensure_ascii=False),flush=True)
            native=batch=prefix=suffix=clone=owners=capq=None; pre.clear()
            for name in ('k','v','kr','ck','cv','q','allk','allv','tk','tv','fields','image_cap','qcap','owner'):
                if name in locals(): locals()[name]=None
            gc.collect(); torch.cuda.empty_cache()
        if case['status']=='FAIL': break
    manifest.update(status='PASS_DIAGNOSTIC_ONLY' if len(completed)==32 and all(v['status'].startswith('PASS') for v in completed) else 'FAIL',
        completed_sources=len(completed),sources=completed,wall_seconds=time.perf_counter()-started)
    write(output/'complete.json',manifest)
    return 0 if manifest['status'].startswith('PASS') else 2

if __name__=='__main__': raise SystemExit(main())
