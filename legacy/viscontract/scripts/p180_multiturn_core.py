"""Single-sequence native cache control and source-derived visual compression.

SnapKV and LOOK-M operators are adapted to visual-only GQA caches. They are not
the original repositories' full streaming implementations.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import math
import torch
import torch.nn.functional as F


def causal_bias(query, key):
    q, k = query.shape[-2], key.shape[-2]
    if q > k or query.shape[0] != 1:
        raise ValueError('single unpadded sequence with all current keys required')
    visible = torch.arange(k, device=query.device)[None] <= torch.arange(k-q, k, device=query.device)[:, None]
    return torch.zeros((1, 1, q, k), device=query.device, dtype=query.dtype).masked_fill_(~visible, float('-inf'))


def grouped_scores(scores, kv_heads):
    b, heads, n = scores.shape
    if heads % kv_heads:
        raise ValueError('invalid GQA head grouping')
    return scores.reshape(b, kv_heads, heads//kv_heads, n).mean(2)


def prefill_scores(query, key, visual, window=64, chunk=32):
    """Full causal attention mass plus SnapKV last-window pooled mass."""
    nq, nk = query.shape[-2], key.shape[-2]
    if nq != nk:
        raise ValueError('score collection requires initial full prefill')
    expanded = key.float().repeat_interleave(query.shape[1]//key.shape[1], 1)
    total = torch.zeros(query.shape[:2]+(nk,), device=query.device, dtype=torch.float32)
    tail = torch.zeros_like(total)
    for start in range(0, nq, chunk):
        end = min(nq, start+chunk)
        logits = query[:, :, start:end].float() @ expanded.transpose(-1, -2) / math.sqrt(query.shape[-1])
        mask = torch.arange(nk, device=query.device)[None] > torch.arange(start, end, device=query.device)[:, None]
        weights = logits.masked_fill(mask, float('-inf')).softmax(-1)
        total += weights.sum(2)
        if end > nq-window:
            tail += weights[:, :, max(0, nq-window-start):].sum(2)
    # The recent window is retained by SnapKV. Pool only the older prefix.
    before = max(0, nq-window)
    pooled = torch.zeros_like(tail)
    if before:
        pooled[..., :before] = F.avg_pool1d(tail[..., :before], 5, 1, 2)
    return {'lookm': grouped_scores(total, key.shape[1]).detach(),
            'snapkv': grouped_scores(pooled, key.shape[1]).detach(),
            'recent_start': before, 'visual': visual}


@contextmanager
def attention_bridge(model, visual=None, tau=0., capture=None, native=False):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    modules = {id(layer.self_attn) for layer in model.model.language_model.layers}

    def apply(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if id(module) not in modules:
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        if native:
            result = original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
            if capture is not None:
                capture[int(module.layer_idx)] = prefill_scores(query, key, visual)
            return result
        # A single decode query sees every cached key. Keep native mask-free
        # SDPA dispatch instead of introducing an all-zero additive mask.
        if query.shape[-2] == 1 and not tau:
            kwargs['is_causal'] = False
            return original(module, query, key, value, None, dropout=dropout, scaling=scaling, **kwargs)
        bias = causal_bias(query, key)
        if tau:
            if visual is None:
                raise ValueError('CQ score calibration requires physical visual indices')
            kv = key.index_select(2, visual).float().repeat_interleave(query.shape[1]//key.shape[1], 1)
            scores = query.float() @ kv.transpose(-1, -2) * (scaling or query.shape[-1]**-.5)
            delta = -tau*(scores-scores.amin(-1, keepdim=True))/(scores.amax(-1, keepdim=True)-scores.amin(-1, keepdim=True)+1e-8)
            bias = bias.expand(1, query.shape[1], -1, -1).clone()
            bias[..., visual] += delta.to(query.dtype)
        kwargs['is_causal'] = False
        return original(module, query, key, value, bias, dropout=dropout, scaling=scaling, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register('sdpa', apply)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


def pivot_merge(key, value, keep):
    """LOOK-M pivot operator, including its include_self=True mean reduction."""
    b, h, n, d = key.shape
    gather = keep[..., None].expand(b, h, -1, d)
    k, v = key.gather(2, gather), value.gather(2, gather)
    mask = torch.ones((b, h, n), device=key.device, dtype=torch.bool).scatter_(2, keep, False)
    dropped_k = key[mask].reshape(b, h, -1, d)
    dropped_v = value[mask].reshape(b, h, -1, d)
    if not dropped_k.shape[2]:
        return k, v
    similarity = F.normalize(dropped_k.float(), dim=-1) @ F.normalize(k.float(), dim=-1).transpose(-1, -2)
    indices = similarity.argmax(-1)[..., None].expand_as(dropped_k)
    merged_k = (dropped_k+k.gather(2, indices))/2
    merged_v = (dropped_v+v.gather(2, indices))/2
    return (k.scatter_reduce(2, indices, merged_k, reduce='mean', include_self=True),
            v.scatter_reduce(2, indices, merged_v, reduce='mean', include_self=True))


def compress_select(source, visual, scores, method, ratio):
    if method not in ('snapkv', 'lookm') or not 0 < ratio <= 1:
        raise ValueError('invalid selection method/budget')
    out = SimpleNamespace(key_cache=[], value_cache=[])
    indices, kept_counts = [], []
    full_bytes = sum(t.numel()*t.element_size() for pair in zip(source.key_cache, source.value_cache) for t in pair)
    visual_bytes = sum((k.shape[0]*k.shape[1]*visual.numel()*k.shape[-1]*k.element_size()*2) for k in source.key_cache)
    for layer, (key, value) in enumerate(zip(source.key_cache, source.value_cache)):
        count = max(1, math.ceil(ratio*visual.numel()))
        score = scores[layer][method].index_select(-1, visual).clone()
        if method == 'snapkv':
            recent = visual >= scores[layer]['recent_start']
            # Preserve all observed recent visual positions; expose any budget excess.
            count = max(count, int(recent.sum()))
            score[..., recent] = float('inf')
        choice = torch.argsort(score, dim=-1, descending=True, stable=True)[..., :count]
        chosen = visual[choice]
        is_text = torch.ones(key.shape[2], device=key.device, dtype=torch.bool)
        is_text[visual] = False
        text = torch.nonzero(is_text).flatten()[None, None].expand(key.shape[0], key.shape[1], -1)
        keep = torch.cat((chosen, text), -1).sort(-1).values
        if method == 'lookm':
            k, v = pivot_merge(key, value, keep)
        else:
            gather = keep[..., None].expand(*keep.shape, key.shape[-1])
            k, v = key.gather(2, gather), value.gather(2, gather)
        out.key_cache.append(k.contiguous())
        out.value_cache.append(v.contiguous())
        indices.append(keep.to(torch.int32))
        kept_counts.append(count)
    resident = sum(t.numel()*t.element_size() for pair in zip(out.key_cache, out.value_cache) for t in pair)
    metadata = sum(i.numel()*i.element_size() for i in indices)
    return out, {'visual_retained': kept_counts, 'visual_original': visual.numel(),
                 'kv_tensor_bytes': resident, 'index_bytes': metadata,
                 'total_cache_ratio': (resident+metadata)/full_bytes,
                 'visual_payload_ratio': (resident-(full_bytes-visual_bytes))/visual_bytes,
                 'target_visual_ratio': ratio,
                 'text_content_may_be_merged': method == 'lookm'}, indices


def next_suffix(processor, question):
    marker = 'P180_BOUNDARY_SENTINEL_779381'
    messages = [{'role': 'assistant', 'content': marker}, {'role': 'user', 'content': question}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, suffix = rendered.split(marker, 1)
    eos = processor.tokenizer.eos_token
    if not suffix.startswith(eos):
        raise ValueError('unsupported chat template closing boundary')
    # EOS has already been consumed into the live cache. Never retokenize answers.
    return processor.tokenizer(suffix[len(eos):], add_special_tokens=False, return_tensors='pt')['input_ids']
