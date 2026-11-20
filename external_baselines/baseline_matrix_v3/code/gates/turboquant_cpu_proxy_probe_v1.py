"""CPU mechanism probe on a frozen CalibQuant C2 diagnostic capture.

The visual K/V here are reconstructed C2 proxies, not the original FP16 visual
states. This run produces no task scores and cannot establish native-model
TurboQuant quality. It checks source identities and exposes numerical structure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from methods.calibquant_adapter import unpack_codes
    from methods.turboquant_adapter import TurboQuantAdapter
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_available(), 'This probe must remain CPU-only'
    torch.set_grad_enabled(False); torch.set_num_threads(2)
    capture = torch.load(args.capture, map_location='cpu', weights_only=False)
    source = capture['owner']; bits = capture['bits']
    k = (source['k_scale'] * (unpack_codes(source['k_packed'], bits).to(source['k_scale'].dtype) - source['k_zero'])).float()
    v = (source['v_scale'] * (unpack_codes(source['v_packed'], bits).to(source['v_scale'].dtype) - source['v_zero'])).float()
    q = capture['q'].float(); tk = capture['text_keys'].float(); tv = capture['text_values'].float()
    adapter = TurboQuantAdapter(seed=42 + 6 * 26)
    owner = adapter.encode(k, v, positions=capture['visual_positions'], layer_idx=26)
    kh, vh = owner.materialize(torch.float32)
    b, hq, nq, d = q.shape; hkv = k.shape[1]; groups = hq // hkv
    scale = capture['scaling'] if capture['scaling'] is not None else d ** -.5
    def qk(keys):
        return (q.reshape(b, hkv, groups, nq, d) @ keys.unsqueeze(2).transpose(-1, -2)).reshape(b, hq, nq, -1) * scale
    def summary(x):
        x = x.float().flatten()
        return dict(mean=float(x.mean()), min=float(x.min()), median=float(x.median()), max=float(x.max()))
    def cosine(a, b): return torch.nn.functional.cosine_similarity(a, b, dim=-1)
    def relative_error(a, b): return float((a - b).norm() / a.norm().clamp_min(1e-20))
    def contrast_metrics(original, approximate):
        delta = approximate - original
        centered = original - original.mean(-1, keepdim=True)
        delta_centered = delta - delta.mean(-1, keepdim=True)
        corr = cosine(centered, approximate - approximate.mean(-1, keepdim=True))
        return dict(logit_relative_L2=relative_error(original, approximate),
            per_head_query_logit_correlation=summary(corr),
            per_head_query_absolute_bias=summary(delta.mean(-1).abs()),
            centered_noise_to_signal=summary(delta_centered.norm(dim=-1) / centered.norm(dim=-1).clamp_min(1e-20)))
    source_dense_scores = owner.codec.attention_score(q.reshape(b,hkv,groups,nq,d), owner.key_quantized._replace(
        mse_indices=owner.key_quantized.mse_indices.unsqueeze(2), qjl_signs=owner.key_quantized.qjl_signs.unsqueeze(2),
        residual_norms=owner.key_quantized.residual_norms.unsqueeze(2), norms=owner.key_quantized.norms.unsqueeze(2)))
    source_dense_scores = source_dense_scores.reshape(b,hq,nq,-1) * scale
    adapter_dense_scores = owner.read_visual_logits(q, scale, 'torch_dequant')
    source_score_error = float((source_dense_scores - adapter_dense_scores).abs().max())
    proxy_scores = qk(k); quant_scores = qk(kh)
    torch.testing.assert_close(source_dense_scores, quant_scores, rtol=2e-5, atol=2e-3)
    qp = capture['query_positions']
    vp = capture['visual_positions']; tp = capture['text_positions']
    positions = torch.cat((vp, tp))
    visible = positions[None,:] <= qp[:,None]
    text_scores = qk(tk)
    def attention(visual_logits, visual_values):
        logits = torch.cat((visual_logits, text_scores), dim=-1).masked_fill(~visible[None,None], -torch.inf)
        p = torch.softmax(logits, -1)
        values = torch.cat((visual_values, tv), -2)
        output = (p.reshape(b,hkv,groups,nq,-1) @ values.unsqueeze(2)).reshape(b,hq,nq,d)
        return output, p
    expected, p = attention(proxy_scores, v)
    k_only, pk = attention(quant_scores, v)
    v_only, pv = attention(proxy_scores, vh)
    both, pb = attention(quant_scores, vh)
    mean_k = k.mean(-2, keepdim=True)
    result = {'status':'CPU_PROXY_MECHANISM_PROBE_COMPLETE',
        'scope':'Frozen real C2-branch query/exact text KV; visual K/V reconstructed from Calib2 owner. NOT original FP16 visual K/V.',
        'GPU_used':False,'task_quality_measured':False,'native_FP16_visual_states_available':False,
        'capture_sha256':hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        'source_turbo_adapter_sha256':hashlib.sha256((ROOT/'code/methods/turboquant_adapter.py').read_bytes()).hexdigest(),
        'shapes':{name:list(t.shape) for name,t in [('q',q),('visual_k_proxy',k),('text_k',tk)]},
        'layer_idx':26, 'codec_seed':42+7*26,
        'source_attention_score_vs_dense_decode_max_abs_error':source_score_error,
        'K_proxy_mean_vector_energy_fraction':float((mean_k.square().sum()*k.shape[-2])/k.square().sum()),
        'K_cosine':summary(cosine(k,kh)), 'K_norm_ratio':summary(kh.norm(dim=-1)/k.norm(dim=-1).clamp_min(1e-20)),
        'K_relative_L2':relative_error(k,kh),'V_cosine':summary(cosine(v,vh)), 'V_relative_L2':relative_error(v,vh),
        'K_quantization_noise_to_across_token_key_contrast':float((kh-k).norm()/(k-mean_k).norm().clamp_min(1e-20)),
        'logit_probe':contrast_metrics(proxy_scores,quant_scores),
        'attention_output_relative_L2':{'K_only':relative_error(expected,k_only), 'V_only':relative_error(expected,v_only), 'both':relative_error(expected,both)},
        'mean_visual_probability_mass':{'proxy_reference':float(p[...,:k.shape[-2]].sum(-1).mean()),
                                         'K_quantized':float(pk[...,:k.shape[-2]].sum(-1).mean())},
        'next_requirement':'Repeat on native FP16 visual K/V and native queries across several real layers; this proxy does not establish root cause.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':main()
