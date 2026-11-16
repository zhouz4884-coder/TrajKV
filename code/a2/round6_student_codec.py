"""Round6 shared, frozen-code materialization for offline native-cache controls.

This module is a calibration reference, never a serving bank.  Serving continues
to own packed codes and F16 amplitudes only.  K is differentiated in the original
native-cache coordinates; the actual packed reader additionally rounds WHT(q).
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from affine_rate_bank import fold
from fast_wht import torch_wht, wht
from fused_progressive import decode_last_generic
from quantizers import binary_centroids, pack1
from calibrate_gradient_rank_round5 import make_prefix_cache, selected_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings_from_config(c):
    s = dict(c.get('mixed_bank_settings', c.get('bank_settings', {})))
    s.setdefault('direction_book_path', c.get('encoder_book_path'))
    s.setdefault('metadata_factor_path', c.get('fixed_metadata_factor_path'))
    s.setdefault('v_group_size', c.get('value_group_size', 'all'))
    s.setdefault('budget_bits', c.get('budget_bits', 1.8))
    if 'selected_units' not in s and 'selected_units' in c:
        s['selected_units'] = c['selected_units']
    if not s.get('selected_units') and not s.get('interaction_policy_path'):
        raise ValueError('The current mixed incumbent priority/policy must be explicit')
    return s


def mask_policy_signature(settings):
    """Only allocation-relevant settings; names/control labels do not affect it."""
    payload = {'selected_units': settings.get('selected_units', []),
        'interaction_policy_strategy': settings.get('interaction_policy_strategy'),
        'budget_bits': float(settings.get('budget_bits', 1.8)),
        'v_group_size': settings.get('v_group_size', 'all'),
        'residual_units_limit': settings.get('res_units_limit', settings.get('residual_units_limit'))}
    if settings.get('interaction_policy_path'):
        payload['interaction_policy_sha256'] = sha(settings['interaction_policy_path'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select_units(settings, base, nl, nh, nv, extra_encoder_bytes=0):
    """The exact MixedResidualBank complete-owner arithmetic, plus new constants."""
    dim = 128
    vg = nv if settings.get('v_group_size', 'all') == 'all' else int(settings['v_group_size'])
    if nv % 128 or vg not in (128, nv):
        raise ValueError('D128/complete g128 or whole-image V groups required')
    full_bytes = nl * nh * nv * dim * 4
    base_bytes = (nl * nh * nv * (32 + 2 + 16) + nl * nh * dim * 2
                  + 4 * nl * nh * dim * (nv // vg) + 256 * 4 * 2
                  + nl * 2 * nh * 4 + 8 * nv)
    constants = 4 * int(base['parameter_count']) + int(extra_encoder_bytes)
    requested = [tuple((str(u[0]).lower(), int(u[1]), int(u[2]))) for u in settings.get('selected_units', [])]
    policy = None
    if settings.get('interaction_policy_path'):
        artifact = torch.load(settings['interaction_policy_path'], map_location='cpu', weights_only=False)
        strategy = settings['interaction_policy_strategy']
        aliases = {'dp': 'dp_exact', 'dp_exact': 'dp'}
        if strategy not in artifact['policies']:
            strategy = aliases.get(strategy, strategy)
        policy = artifact['policies'][strategy]
        constants += policy.numel() * policy.element_size()
        maximum = policy.shape[0] - 1
    else:
        if len(set(requested)) != len(requested):
            raise ValueError('Duplicate incumbent units')
        constants += 2 * (2 * nl * nh)
        maximum = len(requested)
    unit_bytes = nv * (dim // 8 + 2)
    allowed = math.floor(float(settings.get('budget_bits', 1.8)) * full_bytes / 16)
    fixed = base_bytes + constants
    if fixed > allowed:
        raise ValueError('Complete mixed base and encoder constants exceed budget')
    count = min(maximum, (allowed - fixed) // unit_bytes)
    cap = settings.get('res_units_limit', settings.get('residual_units_limit'))
    if cap is not None:
        count = min(count, max(0, int(cap)))
    if policy is None:
        chosen = requested[:count]
    else:
        chosen = []
        for layer, value in enumerate(policy[count]):
            for head in range(nh):
                mode = (int(value) >> (2 * head)) & 3
                for bit, axis in ((1, 'k'), (2, 'v')):
                    if mode & bit:
                        chosen.append((axis, layer, head))
        if len(chosen) != count:
            raise ValueError('Policy row cost mismatch')
    if any(axis not in ('k', 'v') or not 0 <= layer < nl or not 0 <= head < nh for axis, layer, head in chosen):
        raise ValueError('Invalid incumbent units')
    return chosen, {'selected_units': [list(u) for u in chosen], 'residual_units': count,
        'complete_bytes': fixed + count * unit_bytes, 'encoder_constant_bytes': constants,
        'gamma_encoder_constant_bytes': int(extra_encoder_bytes), 'full_visual_bytes': full_bytes,
        'actual_bits': 16 * (fixed + count * unit_bytes) / full_bytes, 'unit_bytes': unit_bytes,
        'budget_bits': float(settings.get('budget_bits', 1.8))}


def signs(words, shape):
    bits = ((words.long()[..., None] >> torch.arange(32, device=words.device)) & 1).reshape(shape)
    return 2 * bits.float() - 1


@torch.no_grad()
def encode_layer(source_k, source_v, book, base, layer, value_group):
    """Exact encoder arithmetic, including F16 base, sign words and MAD alpha."""
    k = source_k.float()
    mean = k.mean(-2, keepdim=True).half()
    z = wht((k - mean.float()).contiguous())
    norm = z.norm(dim=-1, keepdim=True)
    parts = (z / (norm / 128 ** .5).clamp_min(1e-20)).reshape(-1, 4)
    bf, indices = book.float(), []
    for start in range(0, len(parts), 65536):
        part = parts[start:start + 65536]
        distance = part.square().sum(-1, keepdim=True) + bf.square().sum(-1)[None] - 2 * (part @ bf.T)
        indices.append(distance.argmin(-1).byte())
    codes = torch.cat(indices).reshape(*z.shape[:-1], 32)
    template = bf[codes.long()].reshape(z.shape)
    gain = (norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)).half()
    if base.get('key_head_factors') is not None:
        fixed = torch.tensor(base['key_head_factors'][layer], device=k.device)[None, :, None, None]
        gain = (gain.float() * fixed).half()
        if base['key_gain_mode'] == 'whole':
            mean = (mean.float() * fixed).half()
    krot = (template * gain.float()).half()
    ke = z - krot.float()
    record = binary_centroids(source_v.transpose(2, 3).contiguous(), group_size=value_group)
    if base.get('value_head_factors') is not None:
        vg = torch.tensor(base['value_head_factors'][layer], device=k.device)[None, :, None, None]
        vo = torch.tensor(base['value_center_offsets'][layer], device=k.device)[None, :, None, None]
        record = fold(record, 1, vg, vo)
    vd = decode_last_generic(*record, value_group, 1).transpose(2, 3).contiguous()
    ve = source_v.float() - vd.float()
    return {'codes': codes.contiguous(), 'gain': gain.contiguous(), 'mean': mean.contiguous(),
        'krot': krot.contiguous(), 'v_base': vd, 'v_record': record, 'value_group': value_group,
        'k_sign': pack1(ke >= 0), 'k_alpha': ke.abs().mean(-1).half(),
        'v_sign': pack1(ve >= 0), 'v_alpha': ve.abs().mean(-1).half()}


def decoded_layer(entry, layer, selected, gamma=None):
    """Materialize common-coordinate K/V; alpha multiplication rounds into F16."""
    kr, vv = entry['krot'], entry['v_base']
    kh, vh = [], []
    for axis, original, result in (('k', kr, kh), ('v', vv, vh)):
        signed = signs(entry[axis + '_sign'], original.shape)
        for head in range(original.shape[1]):
            value = original[:, head:head + 1]
            if (axis, layer, head) in selected:
                multiplier = 1. if gamma is None else gamma[0 if axis == 'k' else 1, layer, head]
                alpha = (entry[axis + '_alpha'][:, head:head + 1].float() * multiplier).half()
                value = (value.float() + signed[:, head:head + 1] * alpha.float()[..., None]).half()
            result.append(value)
    krot = torch.cat(kh, 1)
    key = (torch_wht(krot.float()) + entry['mean'].float()).half()
    return key, torch.cat(vh, 1)


def make_entries(prefix, encoded, visual, selected, gamma=None):
    entries = []
    for layer, ((key, value), entry) in enumerate(zip(prefix, encoded)):
        key, value = key.detach().clone(), value.detach().clone()
        dk, dv = decoded_layer(entry, layer, selected, gamma)
        key.index_copy_(2, visual, dk)
        value.index_copy_(2, visual, dv)
        entries.append((key, value))
    return entries


@torch.no_grad()
def teacher_case(model, processor, row, a, target_limit, attention_bridge, stop_ids):
    import session_boundary
    features = {k: v.to(model.device, dtype=model.dtype if v.is_floating_point() else v.dtype)
                for k, v in session_boundary.features(processor, row, a.get('max_patches', 8)).items()}
    visual = (features['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
    if not visual.numel():
        raise ValueError('No visual tokens')
    boundary = int(visual[-1]) + 1
    suffix = features['input_ids'][:, boundary:].clone()
    if not suffix.numel():
        raise ValueError('Question suffix must follow image boundary')
    prompt_length = features['input_ids'].shape[-1]
    with attention_bridge(model, native=True):
        out = model(**features, use_cache=True, return_dict=True, logits_to_keep=1)
    cache, ids, logits = out.past_key_values, [], []
    for step in range(target_limit):
        logits.append(out.logits[:, -1].detach().float().cpu())
        token = int(out.logits[:, -1].argmax(-1))
        ids.append(token)
        if token in stop_ids or step + 1 == target_limit:
            break
        with attention_bridge(model, native=True):
            out = model(input_ids=torch.tensor([[token]], device=model.device),
                position_ids=torch.tensor([[prompt_length + step]], device=model.device),
                past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
    prefix = [(layer.keys[:, :, :boundary].detach().clone(), layer.values[:, :, :boundary].detach().clone())
              for layer in cache.layers]
    if len(ids) > 1:
        suffix = torch.cat((suffix, torch.tensor([ids[:-1]], device=model.device)), -1)
    teacher = torch.stack(logits, 1)
    targets = torch.tensor([ids], device=model.device)
    competitors = teacher.clone()
    competitors.scatter_(-1, targets.cpu()[..., None], -torch.inf)
    return {'id': row['id'], 'image_id': row['image_id'], 'prefix': prefix, 'visual': visual,
        'boundary': boundary, 'suffix': suffix, 'targets': targets, 'teacher_logits': teacher,
        'competitors': competitors.argmax(-1).to(model.device), 'prompt_suffix_tokens': prompt_length - boundary,
        'qpos': torch.arange(boundary, boundary + suffix.shape[-1], device=model.device)[None]}


def native_replay(model, case, entries, attention_bridge, requires_grad=False):
    cache, keys, values = make_prefix_cache(entries, model.config.get_text_config(), requires_grad)
    if cache.get_seq_length() != case['boundary']:
        raise RuntimeError('Fresh native prefix length mismatch')
    with attention_bridge(model, native=True):
        out = model(input_ids=case['suffix'], position_ids=case['qpos'], past_key_values=cache,
                    use_cache=True, return_dict=True, logits_to_keep=case['targets'].shape[1])
    return out.logits, keys, values
