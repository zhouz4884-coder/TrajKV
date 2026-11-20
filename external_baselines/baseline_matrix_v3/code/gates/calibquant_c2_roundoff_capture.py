"""One-layer C2 source-kernel replay and roundoff evidence; no model loading.

An operator-only CUDA diagnostic for the root scheduler. It loads an existing
frozen q/owner capture, never reads reference answers, and saves probability and
reduction intermediates for a separate CPU forward-error audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/gates'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':'SOURCE_OPERATOR_ONLY_CUDA_REPLAY_NOT_TASK_QUALITY',
            'model_weights_loaded':False,'references_read':False,'started_unix':time.time()}
    try:
        import torch
        from methods.calibquant_fp32_bias_adapter import CalibQuantFP32BiasState
        from methods.calibquant_adapter import repeat_kv,unpack_codes,_load_upstream
        from cache_bridge import HybridVisualCache
        from calibquant_fp32_bias_oracle import source_dense_attention
        from calibquant_operator_reference import source_operator_attention,difference_summary
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
        torch.cuda.set_device(0);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source=torch.load(args.capture,map_location='cuda:0',weights_only=False)
        q=source['q'];assert source['bits']==2 and q.dtype==torch.float16
        owner=CalibQuantFP32BiasState(**source['owner'],bits=2,reader_backend='upstream_triton')
        vp,tp,qp=source['visual_positions'],source['text_positions'],source['query_positions']
        length=int(torch.cat((vp,tp)).max())+1
        layer=SimpleNamespace(keys=source['text_keys'],values=source['text_values'],visual_owner=owner,
                              visual_positions=vp,text_positions=tp,query_positions=qp,
                              logical_length=length,last_update_length=q.shape[-2])
        cache=SimpleNamespace(layers=[layer],reader_backend='upstream_triton',_mask=HybridVisualCache._mask)
        module=SimpleNamespace(training=False,is_causal=True)
        scale=source['scaling'] if source['scaling'] is not None else q.shape[-1]**-.5
        mask=source['attention_mask']
        actual=HybridVisualCache.read(cache,0,q,module,mask,scale,None)[0]
        assert torch.isfinite(actual).all()
        g=q.shape[1]//owner.k_packed.shape[1];repeat=lambda x:repeat_kv(x,g)
        ks,kz,vs,vz=map(repeat,(owner.k_scale,owner.k_zero,owner.v_scale,owner.v_zero))
        kc,vc=repeat(unpack_codes(owner.k_packed,2)),repeat(unpack_codes(owner.v_packed,2))
        upstream=_load_upstream()
        raw_qk=torch.cat([upstream.fused_kernel(owner.k_lut,(q[:,:,t:t+1]*ks).contiguous(),
                   repeat(owner.k_packed).contiguous(),BITS=2,DATUM_PER_ELM=4) for t in range(q.shape[-2])],2)
        products=ks*kz*q
        assert torch.isfinite(products).all()
        bias_sum=products.sum(-1,keepdim=True,dtype=torch.float32)
        bias=(bias_sum/(1.0/scale)).to(q.dtype)
        uncalibrated=raw_qk*scale-bias
        low,high=uncalibrated.amin(-1,keepdim=True),uncalibrated.amax(-1,keepdim=True)
        calibrated=((uncalibrated-low)/(high-low+1e-8))*(high-3-low)+low
        torch.testing.assert_close(calibrated,owner.read(q,softmax_scale=scale),rtol=0,atol=0)
        text_logits=((q@repeat(layer.keys).transpose(-1,-2))/(1.0/scale)).float()
        visual_logits=calibrated.float()
        logical_mask,standard=HybridVisualCache._mask(mask,q,layer);assert standard
        vm=logical_mask.index_select(-1,vp);tm=logical_mask.index_select(-1,tp)
        if logical_mask.dtype==torch.bool:
            visual_logits=visual_logits.masked_fill(~vm,-torch.inf);text_logits=text_logits.masked_fill(~tm,-torch.inf)
        else:visual_logits=visual_logits+vm;text_logits=text_logits+tm
        joined_logits=torch.cat((visual_logits,text_logits),-1)
        p32=torch.softmax(joined_logits,-1);p=p32.to(q.dtype);w=p[...,:vp.numel()]
        raw_wv=torch.cat([upstream.fused_kernel_wv(owner.v_lut,w[:,:,t:t+1].contiguous(),
                      repeat(owner.v_packed).contiguous(),BITS=2,DATUM_PER_ELM=4) for t in range(q.shape[-2])],2)
        weight_sum=w.sum(-1,keepdim=True)
        affine_left=raw_wv*vs;affine_coefficient=vs*vz;affine_right=affine_coefficient*weight_sum
        visual_output=affine_left-affine_right
        text_output=p[...,vp.numel():]@repeat(layer.values)
        replay=(visual_output+text_output).transpose(1,2).contiguous()
        production_equal=torch.equal(actual,replay)
        captured=[];original_softmax=torch.softmax
        def recording_softmax(*a,**kw):
            result=original_softmax(*a,**kw);captured.append((a[0].detach().clone(),result.detach().clone()));return result
        try:
            torch.softmax=recording_softmax
            ideal_output,ideal_diag=source_dense_attention(cache,0,q,mask,scale)
        finally:torch.softmax=original_softmax
        assert len(captured)==1
        ideal_logits,ideal_p32_original=captured[0]
        order=torch.cat((vp,tp));ideal_p32=ideal_p32_original.index_select(-1,order);ideal_p=ideal_p32.to(q.dtype)
        ideal_original_half=ideal_p32_original.to(q.dtype)
        ideal_w=ideal_original_half.index_select(-1,vp)
        raw_wv_ideal=(ideal_w.double()@vc.double()).to(q.dtype)
        ideal_weight_sum=ideal_w.sum(-1,keepdim=True)
        ideal_visual=raw_wv_ideal*vs-(vs*vz)*ideal_weight_sum
        ideal_text=ideal_original_half.index_select(-1,tp)@repeat(layer.values)
        ideal_replay=(ideal_visual+ideal_text).transpose(1,2).contiguous()
        assert torch.equal(ideal_output,ideal_replay)
        authored,authored_diag=source_operator_attention(cache,0,q,mask)
        raw_qk_ideal=((q*ks).double()@kc.double().transpose(-1,-2)).to(q.dtype)
        raw_wv_same_actual=(w.double()@vc.double()).to(q.dtype)
        payload={key:value.detach().cpu() for key,value in {
            'q':q,'k_scale':ks,'k_zero':kz,'v_scale':vs,'v_zero':vz,'k_codes':kc,'v_codes':vc,
            'text_keys':repeat(layer.keys),'text_values':repeat(layer.values),
            'raw_qk_source':raw_qk,'raw_qk_FP64_rounded':raw_qk_ideal,
            'bias_products_half':products,'bias_sum_float32':bias_sum,'bias_half':bias,
            'uncalibrated_half':uncalibrated,'calibrated_half':calibrated,
            'actual_logits_float32':joined_logits,'actual_probabilities_float32':p32,'actual_probabilities_half':p,
            'ideal_probabilities_float32':ideal_p32,'ideal_probabilities_half':ideal_p,
            'raw_wv_source':raw_wv,'raw_wv_FP64_same_actual_probabilities':raw_wv_same_actual,
            'weight_sum_half':weight_sum,'affine_left_half':affine_left,'affine_coefficient_half':affine_coefficient,
            'affine_right_half':affine_right,'visual_output_half':visual_output,'text_output_half':text_output,
            'actual_output':actual,'replayed_output':replay,'FP64_instrument_output':ideal_output,
            'AST_same_precision_output':authored,'ideal_raw_wv_half':raw_wv_ideal,
            'ideal_weight_sum_half':ideal_weight_sum,'ideal_visual_output_half':ideal_visual,'ideal_text_output_half':ideal_text}.items()}
        payload.update(scaling=scale,visual_tokens=vp.numel(),scope=report['scope'],source_capture_sha256=sha(args.capture))
        evidence=args.output/'operator_intermediates.pt';torch.save(payload,evidence)
        report.update(status='SOURCE_OPERATOR_REPLAY_CAPTURED' if production_equal else 'PRODUCTION_REPLAY_MISMATCH_UNVERIFIED',
            production_source_replay_bitwise_equal=production_equal,
            actual_vs_replay=difference_summary(actual,replay),actual_vs_FP64_instrument=difference_summary(actual,ideal_output),
            actual_vs_author_AST=difference_summary(actual,authored),
            raw_QK_source_vs_FP64=difference_summary(raw_qk,raw_qk_ideal),
            raw_WV_source_vs_FP64_same_actual_probabilities=difference_summary(raw_wv,raw_wv_same_actual),
            actual_vs_old_frozen_output=difference_summary(actual,source['actual_output']),
            source_AST_diagnostics={key:value for key,value in authored_diag.items() if not isinstance(value,torch.Tensor)},
            ideal_instrument_diagnostics=ideal_diag,
            evidence={'file':str(evidence),'sha256':sha(evidence),'bytes':evidence.stat().st_size,'policy':'remote data volume only; no tensor download'},
            source_capture_sha256=sha(args.capture),hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,
            fp16_reduced_precision_reduction_allowed=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            source_sha256={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in
                [Path(__file__),ROOT/'code/methods/calibquant_fp32_bias_adapter.py',ROOT/'code/cache_bridge.py',ROOT/'third_party/calibquant/calibquant.py']})
    except Exception as exc:
        report.update(status='FAILED_RUN',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['completed_unix']=time.time()
        with (args.output/'complete.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
        print(json.dumps({'status':report['status'],'model_weights_loaded':False,'references_read':False}),flush=True)


if __name__=='__main__':main()
