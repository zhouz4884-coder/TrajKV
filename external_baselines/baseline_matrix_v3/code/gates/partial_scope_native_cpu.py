"""CPU fixed-native-query analysis of three frozen OnlyTerp source variants.

No model/GT/new algorithm or parameter fitting. Original native tensors remain
on the data volume; only scalar JSON/CSV is exported. CPU source encoding is an
operator control, not an assertion of byte-identical GPU encodings or E2E scores.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--source-indices',type=int,nargs='+',default=[0]);parser.add_argument('--layers',type=int,nargs='+',default=[0,14,27])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':'CPU_REAL_NATIVE_Q_KV_FIXED_QUERY_OPERATOR_CONTROL',
            'GPU_used':False,'model_weights_loaded':False,'references_or_GT_read':False,'task_quality_scored':False,
            'end_to_end_inference':False,'online_estimator':False,'method_modified':False,'parameters_fitted':False,
            'CPU_encoding_not_asserted_GPU_byte_identical':True,'capture_tensor_download_allowed':False,
            'started_unix':time.time(),'rows':[],'details':[]}
    try:
        import torch
        import torch.nn.functional as F
        from methods.turboquant_onlyterp_mse_adapter import OnlyTerpMSEAdapter,unpack_indices,READER as MSE_READER
        from methods.turboquant_onlyterp_mixed_prod_adapter import OnlyTerpMixedProdAdapter,READER as IP_READER
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        variants=[('MSE_Hmix12',OnlyTerpMSEAdapter(bits_regular=1,bits_outlier=2,mixed_precision=True),MSE_READER),
                  ('MSE_Hmix23',OnlyTerpMSEAdapter(bits_regular=2,bits_outlier=3,mixed_precision=True),MSE_READER),
                  ('IP_Hmix34_singleGaussian',OnlyTerpMixedProdAdapter(bits_regular=3,bits_outlier=4,value_read_mode='source_cache_mse'),IP_READER)]
        report['variants']={name:adapter.manifest() for name,adapter,_ in variants}
        report['torch_version']=torch.__version__
        def stats(value):
            value=value.double().flatten();finite=value[torch.isfinite(value)]
            if not finite.numel():return {'finite':0,'total':value.numel()}
            return {'mean':float(finite.mean()),'mean_abs':float(finite.abs().mean()),'RMS':float(finite.square().mean().sqrt()),
                    'min':float(finite.min()),'median':float(finite.median()),'p95':float(torch.quantile(finite,.95)),
                    'max':float(finite.max()),'finite':finite.numel(),'total':value.numel()}
        def relative(value,reference):return float((value-reference).norm()/reference.norm().clamp_min(1e-30))
        def norm_relative(value,reference):return float(value.norm()/reference.norm().clamp_min(1e-30))
        def tensor_sha(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        def qk(q,k,scale):
            b,hq,nq,d=q.shape;hkv=k.shape[1]
            return ((q.reshape(b,hkv,hq//hkv,nq,d)@k.unsqueeze(2).transpose(-1,-2)).reshape(b,hq,nq,-1))*scale
        def weighted(p,v):
            b,hq,nq,n=p.shape;hkv=v.shape[1];d=v.shape[-1]
            return (p.reshape(b,hkv,hq//hkv,nq,n)@v.unsqueeze(2)).reshape(b,hq,nq,d)
        captures=[]
        for source_index in args.source_indices:
            for layer in args.layers:captures.append((source_index,layer,args.capture_root/f'{source_index:03d}_layer{layer}.pt'))
        report['expected_operator_cases']=len(captures)*len(variants)
        for source_index,layer,path in captures:
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['scope']=='NATIVE_FP16_PREFIX_AND_SUFFIX_Q_KV_NOT_COMPRESSED_BRANCH'
            assert state['layer_idx']==layer
            qh,kh,vh=state['q'],state['visual_k'],state['visual_v']
            assert qh.dtype==kh.dtype==vh.dtype==torch.float16
            q,k,v=qh.double(),kh.double(),vh.double()
            tk,tv=state['text_keys'].double(),state['text_values'].double()
            vp,tp,qp=state['visual_positions'],state['text_positions'],state['query_positions']
            assert int(vp.max())<int(qp.min())
            scale=float(state['scaling']);b,hq,nq,d=q.shape;hkv=k.shape[1];nv=k.shape[-2]
            z0=qk(q,k,scale);zt=qk(q,tk,scale)
            text_visible=tp[None,:]<=qp[:,None]
            zt=zt.masked_fill(~text_visible[None,None],-torch.inf)
            logz0=torch.logsumexp(z0,-1);logzt=torch.logsumexp(zt,-1)
            assert torch.isfinite(logz0).all() and torch.isfinite(logzt).all()
            pi0=(z0-logz0[...,None]).exp();tau=(zt-logzt[...,None]).exp()
            s0=torch.sigmoid(logz0-logzt);m0=weighted(pi0,v);mt=weighted(tau,tv)
            o0=s0[...,None]*m0+(1-s0[...,None])*mt
            kj_mean=k.mean(-2,keepdim=True);native_center=k-kj_mean
            capture_meta={'source_index':source_index,'id':state['id'],'layer':layer,'path':str(path),'sha256':sha(path),
                          'original_image_sha256':state['original_image_sha256'],'q_shape':list(q.shape),'visual_KV_shape':list(k.shape)}
            for name,adapter,backend in variants:
                started=time.perf_counter();owner=adapter.encode(kh,vh,positions=vp,layer_idx=layer)
                kd,vd=owner.materialize()
                assert kd.dtype==vd.dtype==torch.float32,'Source decode must not be silently Half-rounded'
                k1,v1=kd.double(),vd.double()
                zdecoded=qk(q,k1,scale)
                if name.startswith('IP_'):zsource=owner.read_visual_logits(qh,scale).double()
                else:zsource=qk(qh.float(),kd,scale).double()
                assert torch.isfinite(zsource).all()
                e=zsource-z0;beta=e.mean(-1);eta=e-beta[...,None]
                mean_error=(k1-k).mean(-2,keepdim=True)
                betaK=(q.reshape(b,hkv,hq//hkv,nq,d)*mean_error.unsqueeze(2)).sum(-1).reshape(b,hq,nq)*scale
                arithmetic=zsource-zdecoded
                beta_identity=beta-betaK-arithmetic.mean(-1)
                logz1=torch.logsumexp(zsource,-1);pi1=(zsource-logz1[...,None]).exp()
                logmoment=torch.logsumexp(z0-logz0[...,None]+eta,-1)
                delta=logz1-logz0
                logz_identity=delta-beta-logmoment
                s1=torch.sigmoid(logz1-logzt);s_identity=s1-torch.sigmoid(logz0-logzt+beta+logmoment)
                tv_route=(pi1-pi0).abs().sum(-1)/2
                R=eta.amax(-1)-eta.amin(-1);tv_bound=torch.tanh(R/4)
                assert bool((tv_route<=tv_bound+1e-10).all())
                m1K=weighted(pi1,v);m1KV=weighted(pi1,v1)
                oK=s1[...,None]*m1K+(1-s1[...,None])*mt
                oKV=s1[...,None]*m1KV+(1-s1[...,None])*mt
                oV=s0[...,None]*weighted(pi0,v1)+(1-s0[...,None])*mt
                mass=(s1-s0)[...,None]*(m0-mt)
                route=s1[...,None]*(m1K-m0);content=s1[...,None]*(m1KV-m1K)
                identity_K=(oK-o0)-mass-route
                identity_KV=(oKV-o0)-mass-route-content
                actual=owner.attend(qh,exact_keys=state['text_keys'],exact_values=state['text_values'],
                    exact_positions=tp,query_positions=qp,scale=scale,backend=backend).double()
                E=k1-k;centerE=E-mean_error
                delta_center=z0-z0.mean(-1,keepdim=True)
                noise_ratio=eta.norm(dim=-1)/delta_center.norm(dim=-1).clamp_min(1e-30)
                weighted_eta=(pi0*eta).sum(-1)
                positive=E.square().sum().clamp_min(1e-30)
                top_agreement=(z0.argmax(-1)==zsource.argmax(-1)).double()
                detail={'capture':capture_meta,'variant':name,'source_manifest':adapter.manifest(),
                    'beta_score':stats(beta),'beta_reconstructed_K':stats(betaK),'eta':stats(eta),
                    'pi_weighted_eta_mean':stats(weighted_eta),'exact_logmoment_eta':stats(logmoment),
                    'logmoment_minus_pi_mean_eta':stats(logmoment-weighted_eta),'delta_logZ':stats(delta),
                    'native_visual_mass':stats(s0),'quantized_visual_mass':stats(s1),'conditional_routing_TV':stats(tv_route),
                    'eta_range':stats(R),'TV_tanh_R_over4_bound':stats(tv_bound),
                    'source_score_minus_reconstructed_dot':stats(arithmetic),
                    'source_score_arithmetic_mean':stats(arithmetic.mean(-1)),
                    'source_score_arithmetic_centered':stats(arithmetic-arithmetic.mean(-1,keepdim=True)),
                    'source_tensor_metadata':{key:{'dtype':str(value.dtype),'shape':list(value.shape),'sha256':tensor_sha(value)}
                                              for key,value in owner.named_tensors().items()},
                    'identity_residuals':{'beta_from_meanK_plus_arithmetic_max_abs':float(beta_identity.abs().max()),
                        'uniform_eta_mean_max_abs':float(eta.mean(-1).abs().max()),'logZ_max_abs':float(logz_identity.abs().max()),
                        'visual_mass_max_abs':float(s_identity.abs().max()),'output_K_only_max_abs':float(identity_K.abs().max()),
                        'output_KV_max_abs':float(identity_KV.abs().max())},
                    'operator_numeric_output_residual':{'relative_L2':relative(actual,oKV),'max_abs':float((actual-oKV).abs().max()),
                        'scope':'source CPU FP32 reader plus FP16 final return versus fixed-source-visual-score FP64/text-exact decomposition'},
                    'teacher_probes_require_native_references':True,'source_cpu_wall_seconds':time.perf_counter()-started}
                row={'source_index':source_index,'id':state['id'],'layer':layer,'variant':name,'visual_tokens':nv,'query_tokens':nq,
                    'native_K_mean_energy_fraction':float(kj_mean.square().sum()*nv/k.square().sum()),
                    'decoded_K_relative_L2':relative(k1,k),'decoded_V_relative_L2':relative(v1,v),
                    'K_cosine_mean':float(F.cosine_similarity(k,k1,dim=-1).mean()),
                    'K_error_mean_energy_fraction':float(mean_error.square().sum()*nv/positive),
                    'K_mean_error_relative_to_native_mean':float(mean_error.norm()/kj_mean.norm().clamp_min(1e-30)),
                    'K_centered_error_relative_to_native_variation':float(centerE.norm()/native_center.norm().clamp_min(1e-30)),
                    'beta_mean':float(beta.mean()),'beta_RMS':float(beta.square().mean().sqrt()),'beta_K_RMS':float(betaK.square().mean().sqrt()),
                    'eta_RMS':float(eta.square().mean().sqrt()),'centered_logit_noise_signal_mean':float(noise_ratio.mean()),
                    'exact_logmoment_eta_mean':float(logmoment.mean()),'pi_weighted_eta_mean':float(weighted_eta.mean()),
                    'delta_logZ_mean':float(delta.mean()),'native_visual_mass_mean':float(s0.mean()),'quantized_visual_mass_mean':float(s1.mean()),
                    'conditional_TV_mean':float(tv_route.mean()),'conditional_TV_max':float(tv_route.max()),
                    'conditional_visual_top1_agreement':float(top_agreement.mean()),
                    'output_K_only_relative_L2':relative(oK,o0),'output_V_only_relative_L2':relative(oV,o0),'output_KV_relative_L2':relative(oKV,o0),
                    'output_mass_term_norm_relative':norm_relative(mass,o0),'output_route_term_norm_relative':norm_relative(route,o0),
                    'output_value_term_norm_relative':norm_relative(content,o0),
                    'source_FP32_score_vs_decoded_dot_max_abs':float(arithmetic.abs().max()),
                    'source_reader_vs_fixedq_FP64_output_relative_L2':relative(actual,oKV),
                    'logZ_identity_max_abs':float(logz_identity.abs().max()),'output_identity_max_abs':float(identity_KV.abs().max()),
                    'CPU_wall_seconds':detail['source_cpu_wall_seconds']}
                if name.startswith('IP_'):
                    baseK,baseV=owner.mse_owner.materialize()
                    assert baseK.dtype==baseV.dtype==torch.float32
                    residual=kh.float()-baseK
                    source_rnorm=owner.k_qjl_rnorm
                    signs=unpack_indices(owner.k_qjl_signs,1,128,dtype=torch.float64)*2-1
                    S=owner.gaussian_matrices.double();qg=q.reshape(b,hkv,hq//hkv,nq,d)
                    c=math.sqrt(math.pi/2)/128
                    correction_vector=(signs@S)*source_rnorm.double()[...,None]*c
                    idealIP=baseK.double()+correction_vector
                    separate=(qg@baseK.double().unsqueeze(2).transpose(-1,-2))*scale
                    projection=qg@S[None,:,None].transpose(-1,-2)
                    qjl=(projection@signs.unsqueeze(2).transpose(-1,-2))*source_rnorm.double()[:,:,None,None,:]*c*scale
                    direct=qk(q,idealIP,scale)
                    fixed_identity=direct-(separate+qjl).reshape_as(direct)
                    detail['single_Gaussian_IP']={'source_factor':c,'correction_count':1,'Gaussian_shape':list(S.shape),
                        'MSE_base_K_relative_L2':relative(baseK.double(),k),'IP_decoded_K_relative_L2':relative(k1,k),
                        'MSE_residual_norm':stats(residual.norm(dim=-1)),'stored_QJL_residual_norm':stats(source_rnorm),
                        'stored_vs_recomputed_FP32_residual_norm_max_abs':float((source_rnorm-residual.norm(dim=-1)).abs().max()),
                        'correction_vector_norm':stats(correction_vector.norm(dim=-1)),
                        'correction_vs_MSE_residual_cosine':stats(F.cosine_similarity(correction_vector,residual.double(),dim=-1)),
                        'exact_single_projection_dot_identity_max_abs':float(fixed_identity.abs().max()),
                        'FP32_public_decode_vs_exact_single_projection_vector_relative_L2':relative(k1,idealIP),
                        'value_read_mode':owner.value_read_mode,'V_QJL_stored_but_not_read':True}
                    row.update(IP_MSE_base_K_relative_L2=relative(baseK.double(),k),
                        IP_residual_norm_mean=float(source_rnorm.mean()),IP_dot_identity_max_abs=float(fixed_identity.abs().max()))
                # Retain a few head-query scalars, not original tensor features.
                rank=(s1-s0).abs().flatten().topk(min(5,s0.numel())).indices
                worst=[]
                for flat in rank.tolist():
                    query=flat%nq;head=(flat//nq)%hq
                    worst.append({'head':head,'query':query,'native_mass':float(s0[0,head,query]),'quantized_mass':float(s1[0,head,query]),
                        'beta':float(beta[0,head,query]),'beta_K':float(betaK[0,head,query]),
                        'exact_logmoment_eta':float(logmoment[0,head,query]),'delta_logZ':float(delta[0,head,query]),
                        'eta_range':float(R[0,head,query]),'conditional_TV':float(tv_route[0,head,query])})
                detail['largest_mass_change_head_queries']=worst
                report['rows'].append(row);report['details'].append(detail)
                with (args.output/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(detail,ensure_ascii=False)+'\n')
                print(json.dumps({'id':state['id'],'layer':layer,'variant':name,'K_only_relative_L2':row['output_K_only_relative_L2'],
                    'native_mass':row['native_visual_mass_mean'],'quantized_mass':row['quantized_visual_mass_mean'],
                    'CPU_seconds':row['CPU_wall_seconds']}),flush=True)
        report['status']='CPU_FIXED_QUERY_OPERATOR_ANALYSIS_COMPLETE'
        report['source_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
            [Path(__file__),ROOT/'code/methods/turboquant_onlyterp_mse_adapter.py',ROOT/'code/methods/turboquant_onlyterp_mixed_prod_adapter.py',
             ROOT/'configs/methods/onlyterp_mse_source_lock_v2.json',ROOT/'configs/methods/onlyterp_mixed_prod_source_lock_v1.json',
             ROOT/'third_party/turboquant_onlyterp/src/cache.py']}
    except Exception as exc:report.update(status='FAILED_CPU_ANALYSIS',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['completed_unix']=time.time()
        if report['rows']:
            fields=list(dict.fromkeys(key for row in report['rows'] for key in row))
            with (args.output/'summary.csv').open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(report['rows'])
        with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        print(json.dumps({'status':report['status'],'operator_cases':len(report['rows']),'GPU_used':False}),flush=True)


if __name__=='__main__':main()
