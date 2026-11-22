"""Round10: a conditional LS proposal and actual-model TF selection.

No labels, optimizer loop, generation-quality score, remote transport or queue
operations belong here.  Private native tensors/WO stay in the run directory.
Every selection replay starts from an independent, question-free native prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

SCHEMA = 'trajkv.round10.calibration_config.v1'
THETA_SCHEMA = 'trajkv.round10.value_theta.v1'
BASE_PROFILE = 'r7_joint_k1_gt_cond'
BASE_FACTOR_SHA = '8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea'
SUFFIX = '\nAnswer the question using a single word or phrase.'
DOMAINS = {'textvqa', 'infovqa', 'docvqa', 'chartqa'}
LAMBDAS = (0., .25, .5, 1.)
WINDOWS = (1, 32, 64, 128, 256, 'global')
FORBIDDEN = {'answers', 'labels', 'references', 'reference_path', 'reference_answers',
             'development_scores', 'test_scores', 'scores', 'refs'}
PINS = {
    'code/round9_calibration_v1.py': '757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f',
    'code/round9_value_bank_v1.py': '14d242a6643a738457c9b8593d0f0eb6e383c5e979b2e196fcca28e7d37a5f2e',
    'code/r7_legacy_bridge_v1.py': 'b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b',
    'code/r7_legacy_bridge_v2.py': '608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad',
    'code/m1_protocol.py': '4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427',
    'code/cache_bridge.py': '36b860522276895dd361120d315d745a785fd020cbddbff9e75051e26a907025',
    'code/attention_bridge.py': 'eef35d19dfc04fab12f0b6f9ba618a8b34ed48f4488edd566ea4578fb3311876',
    'code/ownership.py': '8443adca9e991663e42da6eff0609b54eb7ef0b13fee8ac09f4e12d1894a8456',
    'code/r7_frozen_pilot_v2.py': '3ff896405be5ac91067bdbe8d736c0552e46f835dea90a93c9835ceeb8fd3a91',
    'code/model_registry.py': '969b39c1df27b63166f4b783187084022cf8548799a6c08178b8665b54d13c58',
    'code/evaluate.py': 'dbb556d87827c146e596b1e01de6c58a7a8b08f788c4c13d18129f4b1f13f75c',
    'code/round10_value_bank_v1.py': '5c9880d9e7facdc1623473dd74759aee9064bd5ff313f1581390fe0ccd3aae1c',
}


def require(value, message):
    if not value:
        raise ValueError('ROUND10_CALIBRATION: ' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def tensor_sha(value):
    tensor = value.detach().cpu().contiguous()
    return canonical({'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                      'stride': list(value.stride()),
                      'bytes_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('purpose') == 'calibration_only', 'explicit calibration scope')
    require(not FORBIDDEN.intersection(config), 'no labels or reference paths')
    require(config.get('model_id') == 'M1' and config.get('seed', 0) == 0, 'M1 seed0 only')
    require(config.get('question_suffix') == SUFFIX and config.get('preprocessing', {}).get('max_patches') == 8,
            'frozen cap8/question suffix')
    require(config.get('teacher_max_new_tokens', 32) == 32, 'teacher EOS/cap32')
    require(config.get('fit_capture_decode_steps', 2) == 2, 'explicit suffix-last plus first2 fit states')
    require(config.get('variant_source', {}).get('name') == BASE_PROFILE, 'original low R7 profile only')
    for role in ('fit', 'select'):
        require(isinstance(config.get(f'cal_{role}_inputs'), str) and
                len(config.get(f'cal_{role}_sha256', '')) == 64, 'locked ' + role + ' inputs')
    expected = config.get('expected_source_hashes', {})
    require(isinstance(expected, dict) and
            {'code/round10_inputs_v1.py', 'code/round10_value_bank_v1.py',
             'code/round10_value_bank_v2.py'}.issubset(expected),
            'pin new loader and factory')
    for key, digest in expected.items():
        require(key.startswith('code/') and '..' not in Path(key).parts and len(digest) == 64,
                'finite project source lock')
    require(config.get('ls_ridge', 1e-5) > 0 and config.get('gain_floor', 1e-4) > 0, 'positive LS regularization/floor')
    return config


def check_sources(config):
    paths = set(PINS) | set(config['expected_source_hashes']) | {
        'code/round10_calibrate_v1.py', 'code/m1_protocol.py', 'code/cache_bridge.py',
        'code/attention_bridge.py', 'code/ownership.py', 'code/r7_frozen_pilot_v2.py',
        'code/model_registry.py', 'code/evaluate.py'}
    result = {key: sha(ROOT / key) for key in sorted(paths)}
    for locks in (PINS, config['expected_source_hashes']):
        for key, digest in locks.items():
            require(result[key] == digest, 'source lock differs: ' + key)
    return result


def validate_rows(fit_rows, select_rows):
    for rows, count, per_domain in ((fit_rows, 48, 12), (select_rows, 16, 4)):
        require(len(rows) == count and Counter(r['dataset'] for r in rows) ==
                Counter({d: per_domain for d in DOMAINS}), 'four-domain calibration allocation')
        require(len({r['original_image_sha256'] for r in rows}) == count, 'one image per calibration source')
        require(len({(r['dataset'], r['source_id']) for r in rows}) == count, 'unique sources')
        require(all(not FORBIDDEN.intersection(r) for r in rows), 'answer-free input rows')
    for key in ('original_image_sha256', 'source_cluster_id'):
        require(not {r[key] for r in fit_rows}.intersection(r[key] for r in select_rows),
                'fit/select overlap: ' + key)
    require(not {(r['dataset'], r['id']) for r in fit_rows}.intersection(
        (r['dataset'], r['id']) for r in select_rows), 'fit/select ID overlap')


def interpolate_theta(base, proposal_g, proposal_h, coefficient):
    """Interpolate absolute (g,h=g*offset), never interpolate offset itself."""
    import torch
    require(coefficient in LAMBDAS, 'declared lambda grid')
    g0 = torch.as_tensor(base['value_head_factors'], dtype=torch.float32)
    o0 = torch.as_tensor(base['value_center_offsets'], dtype=torch.float32)
    g1, h1 = torch.as_tensor(proposal_g, dtype=torch.float32), torch.as_tensor(proposal_h, dtype=torch.float32)
    require(g0.shape == o0.shape == g1.shape == h1.shape == (28, 4), 'absolute V224 geometry')
    require(bool(torch.isfinite(g0).all() & torch.isfinite(o0).all() & torch.isfinite(g1).all() & torch.isfinite(h1).all()),
            'finite interpolation')
    if coefficient == 0:
        # Preserve original offsets, avoiding a multiply/divide identity roundtrip.
        g, offset = g0.clone(), o0.clone()
    else:
        g = g0 + coefficient * (g1 - g0)
        h = g0 * o0 + coefficient * (h1 - g0 * o0)
        offset = h / g
    require(bool((g > 0).all() & torch.isfinite(g).all() & torch.isfinite(offset).all()), 'positive finite absolute theta')
    return {'value_head_factors': g, 'value_center_offsets': offset}


def theta_document(theta, provenance):
    return {'schema': THETA_SCHEMA, 'base_profile': BASE_PROFILE,
            'base_metadata_factor_sha256': BASE_FACTOR_SHA, 'parameter_count': 224,
            **{key: value.detach().cpu().tolist() for key, value in theta.items()},
            'fit_provenance': provenance}


def full_vocab_kl(teacher, student):
    """Actual GPU logits, stable CPU float64 arithmetic; no CUDA F64 log_softmax."""
    import torch
    require(teacher.shape == student.shape and teacher.ndim == 2 and teacher.shape[0] > 0,
            'same teacher-forced positions and vocabulary')
    t, s = teacher.to(device='cpu', dtype=torch.float64), student.to(device='cpu', dtype=torch.float64)
    require(bool(torch.isfinite(t).all() & torch.isfinite(s).all()), 'finite raw logits')
    def lp(x):
        shifted = x - x.amax(-1, keepdim=True)
        result = shifted - shifted.exp().sum(-1, keepdim=True).log()
        require(bool((result.exp().sum(-1) - 1).abs().max() <= 1e-10), 'stable KL probability normalization')
        return result
    lt, ls = lp(t), lp(s)
    rows = (lt.exp() * (lt - ls)).sum(-1)
    require(bool(torch.isfinite(rows).all()) and float(rows.min()) >= -1e-10, 'invalid full vocabulary KL')
    return {'source_mean_KL': float(rows.mean()), 'position_KL': rows.tolist(),
            'positions': len(rows), 'normalization': 'CPU float64 explicit max-shift-exp-sum-log',
            'raw_logits_not_generation_processor_scores': True}


def select_common_lambda(losses, expected_sources):
    require(set(losses) == set(LAMBDAS), 'complete lambda grid including identity control')
    values = {}
    for coefficient in LAMBDAS:
        branches = losses[coefficient]
        require(set(branches) == {'LS-FZ', 'LS-RE'}, 'both deployments at every lambda')
        means = {}
        for mode, rows in branches.items():
            require(set(rows) == set(expected_sources), 'complete source-balanced lambda coverage')
            require(all(math.isfinite(v) and v >= -1e-10 for v in rows.values()), 'finite actual KL')
            means[mode] = sum(rows.values()) / len(expected_sources)
        values[coefficient] = {**means, 'common_objective': .5 * sum(means.values())}
    selected = min((.25, .5, 1.), key=lambda x: (values[x]['common_objective'], x))
    return selected, values


def select_window(losses, expected_sources):
    require(set(losses) == set(WINDOWS), 'all declared window rules including W1')
    values = {}
    for window, rows in losses.items():
        require(set(rows) == set(expected_sources), 'complete window source coverage')
        require(all(math.isfinite(v) and v >= -1e-10 for v in rows.values()), 'finite window KL')
        values[window] = sum(rows.values()) / len(expected_sources)
    selected = min(WINDOWS[1:], key=lambda x: (values[x], math.inf if x == 'global' else x))
    return selected, values


def state_signatures(owners):
    """Reader fields only; theta asset bytes are separately charged by ownership."""
    import torch
    full, k_only = [], []
    for owner in owners:
        state = owner.state
        tensors = {k: tensor_sha(v) for k, v in vars(state).items()
                   if isinstance(v, torch.Tensor) and k not in ('text_k', 'text_v', 'text_pos')}
        full.append(tensors)
        k = {name: tensors[name] for name in ('direction_code', 'direction_gain', 'direction_mean')}
        lookup = state.depth_residual_lookup
        k['lookup'] = tensor_sha(lookup[0])
        for head in range(4):
            for stage in range(2):
                slot = int(lookup[0, head, stage])
                if slot >= 0:
                    k[f'K_{head}_{stage}_code'] = tensor_sha(state.depth_residual_code[slot])
                    k[f'K_{head}_{stage}_alpha'] = tensor_sha(state.depth_residual_alpha[slot])
        k_only.append(k)
    return {'reader_field_sha256': canonical(full), 'K_fields_sha256': canonical(k_only),
            'per_layer_reader_fields': full}


def first_call_guard(model, batch, cache):
    """Context manager: observe true model kwargs, not merely intended inputs."""
    import torch
    from contextlib import contextmanager
    @contextmanager
    def guard():
        events = []
        def before(module, args, kwargs):
            if events:
                return
            ids = kwargs.get('input_ids', args[0] if args else None)
            past = kwargs.get('past_key_values')
            require(past is cache and int(past.get_seq_length()) == batch.boundary, 'first call did not start at question-free boundary')
            require(ids is not None and torch.equal(ids, batch.ids[:, batch.boundary:]), 'first call must recompute original question suffix')
            positions = kwargs.get('position_ids')
            if positions is not None:
                require(torch.equal(positions, batch.positions[:, batch.boundary:]), 'first actual positions changed')
            cp = kwargs.get('cache_position')
            if cp is not None:
                require(torch.equal(cp, torch.arange(batch.boundary, batch.full_length, device=cp.device)), 'first cache positions changed')
            events.append({'actual_input_ids_sha256': tensor_sha(ids), 'cache_tokens_before': batch.boundary,
                           'position_ids_exposed': positions is not None, 'cache_position_exposed': cp is not None})
        handle = model.register_forward_pre_hook(before, with_kwargs=True)
        try:
            yield events
            require(len(events) == 1, 'missing first-call identity observation')
        finally:
            handle.remove()
    return guard()


def eos_ids(model):
    value = model.generation_config.eos_token_id
    return set(value if isinstance(value, list) else [value]) - {None}


def teacher_history(model, batch, native):
    import torch
    from m1_protocol import clone_m1_prefix_cache, m1_generate_cached
    cache = clone_m1_prefix_cache(native)
    logits = []
    def after(module, args, kwargs, result):
        value = getattr(result, 'logits', None)
        require(value is not None and bool(torch.isfinite(value).all()), 'nonfinite teacher outer logits')
        logits.append(value[0, -1].detach().float().cpu().clone())
    handle = model.register_forward_hook(after, with_kwargs=True)
    try:
        with first_call_guard(model, batch, cache) as identity:
            sequence = m1_generate_cached(model, batch, cache, max_new_tokens=32)
        require(torch.equal(sequence[:, :batch.full_length], batch.ids), 'teacher generation returned different prompt')
        tokens = sequence[0, batch.full_length:].cpu().tolist()
        stop = eos_ids(model)
        ends = [i + 1 for i, token in enumerate(tokens) if token in stop]
        if ends:
            tokens = tokens[:ends[0]]
        require(0 < len(tokens) <= 32 and len(logits) == len(tokens), 'teacher EOS/cap/raw-logits alignment')
        return {'token_ids': tokens, 'raw_logits': torch.stack(logits), 'first_call': identity[0],
                'EOS_reached': tokens[-1] in stop, 'max_new_tokens': 32,
                'teacher_history_is_actual_native_generate': True}
    finally:
        handle.remove()


def feed_teacher(model, batch, cache, tokens):
    """Yield each live-student distribution at the teacher answer positions."""
    import torch
    from m1_protocol import m1_forward_suffix
    require(tokens and len(tokens) <= 32, 'finite teacher history')
    with first_call_guard(model, batch, cache):
        result = m1_forward_suffix(model, batch, cache)
    require(int(cache.get_seq_length()) == batch.full_length, 'question suffix not written to student cache')
    yield result.logits[0, -1]
    for index, token in enumerate(tokens[:-1]):
        require(token not in eos_ids(model), 'teacher history continued after EOS')
        position = batch.full_length + index
        mask = torch.cat((batch.mask, torch.ones((1, index + 1), dtype=batch.mask.dtype, device=batch.ids.device)), dim=-1)
        result = model(input_ids=torch.tensor([[token]], device=batch.ids.device), attention_mask=mask,
                       position_ids=torch.tensor([[position]], device=batch.ids.device),
                       past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
        require(int(cache.get_seq_length()) == position + 1, 'student teacher-forced cache positions changed')
        yield result.logits[0, -1]


def local_teacher(sample):
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    keys = torch.cat((sample['visual_k'], sample['exact_keys']), dim=2)
    values = torch.cat((sample['visual_v'], sample['exact_values']), dim=2)
    positions = torch.cat((sample['visual_positions'], sample['exact_positions']))
    order = positions.argsort()
    keys = keys.index_select(2, order).repeat_interleave(7, 1).contiguous()
    values = values.index_select(2, order).repeat_interleave(7, 1).contiguous()
    visible = (positions[order][None] <= sample['query_positions'][:, None])[None, None]
    return scaled_dot_product_attention(sample['q'], keys, values, attn_mask=visible,
                                        dropout_p=0., is_causal=False, scale=sample['scaling'])


def build_cache(native, batch, variant, config, *, mode=None, theta=None, window=None):
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from r7_frozen_pilot_v2 import share_original_visual_position_map
    clone = clone_m1_prefix_cache(native)
    kwargs = {key: config[key] for key in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source')}
    kwargs['expected_asset_sha256'] = config['expected_asset_sha256']
    if mode is None:
        from r7_legacy_bridge_v2 import build_r7_owners
        owners, metadata = build_r7_owners(clone, batch.visual_positions, variant, **kwargs)
    else:
        from round10_value_bank_v2 import build_round10_value_owners
        owners, metadata = build_round10_value_owners(clone, batch.visual_positions, variant, mode=mode,
            theta_override=theta, calibration_only=True, window=window, tile_tokens=256, **kwargs)
    require(len(owners) == 28 and metadata.get('selected_stage_count') == 43, 'unchanged all-layer43 stage scope')
    signatures = state_signatures(owners)
    cache = HybridVisualCache(clone, batch.visual_positions, owners, consume_native=True)
    share_original_visual_position_map(cache)
    return cache, owners, metadata, signatures


def prepare_case(model, processor, row):
    from PIL import Image
    import torch
    from m1_protocol import prepare_m1_inputs, m1_prefix_forward
    require(sha(row['image_path']) == row['original_image_sha256'], 'image SHA differs')
    with Image.open(row['image_path']) as im:
        image = im.convert('RGB')
    batch = prepare_m1_inputs(processor, image, row['question'] + SUFFIX, max_patches=8)
    alternate_question = ('What is visible in the image?' if row['question'] == 'Describe the objects.'
                          else 'Describe the objects.')
    alternate = prepare_m1_inputs(processor, image, alternate_question + SUFFIX, max_patches=8)
    require(batch.boundary == alternate.boundary and torch.equal(batch.ids[:, :batch.boundary], alternate.ids[:, :alternate.boundary])
            and torch.equal(batch.visual_positions, alternate.visual_positions), 'prefix depends on future question')
    for key in set(batch.inputs) | set(alternate.inputs):
        if key in ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids'):
            continue
        a, b = batch.inputs.get(key), alternate.inputs.get(key)
        require(torch.equal(a, b) if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) else a == b,
                'visual processing depends on future question')
    alternate = image = None
    output = m1_prefix_forward(model, batch)
    native = output.past_key_values
    require(int(native.get_seq_length()) == batch.boundary and len(native.layers) == 28, 'question-free native prefix geometry')
    identity = {'id': row['id'], 'dataset': row['dataset'], 'source_id': row['source_id'],
                'image_sha256': row['original_image_sha256'], 'input_ids_sha256': tensor_sha(batch.ids),
                'positions_sha256': tensor_sha(batch.positions), 'visual_positions_sha256': tensor_sha(batch.visual_positions),
                'prefix_boundary': batch.boundary, 'prompt_sha256': hashlib.sha256(batch.prompt.encode()).hexdigest(),
                'question_unknown_prefix_verified': True, 'visual_tokens': int(batch.visual_positions.numel())}
    return batch, native, identity


def capture_fit_case(model, batch, native, row, variant, config, history, initial_factors):
    import torch
    from attention_bridge import attention_context
    cache, owners, metadata, signatures = build_cache(native, batch, variant, config)
    from fast_wht import wht
    from quantizers import binary_centroids
    layers = []
    for i, owner in enumerate(owners):
        state = owner.state
        vk = native.layers[i].keys.index_select(2, batch.visual_positions).detach()
        vv = native.layers[i].values.index_select(2, batch.visual_positions).detach()
        raw = binary_centroids(vv.transpose(2, 3).contiguous(), group_size=int(batch.visual_positions.numel()))
        require(torch.equal(raw[0], state.v_code), 'original raw V assignment changed')
        fields = {key: value.detach().cpu().contiguous().clone() for key, value in vars(state).items()
                  if isinstance(value, torch.Tensor) and key not in ('visual_pos', 'text_k', 'text_v', 'text_pos')}
        n = int(batch.visual_positions.numel())
        fields.setdefault('depth_residual_code', torch.empty((0, n, 4), dtype=torch.int32))
        fields.setdefault('depth_residual_alpha', torch.empty((0, n), dtype=torch.float16))
        layers.append({'visual_k': vk.cpu().clone(), 'visual_v': vv.cpu().clone(), 'r7_fields': fields,
                       'raw_v_code': raw[0].cpu(), 'raw_v_scale': raw[1].cpu(), 'raw_v_min': raw[2].cpu()})
    events, phase = [], {'step': 0}
    originals = []
    for i, owner in enumerate(owners):
        original = owner.attend
        originals.append((owner, original))
        def wrapped(q, *, _index=i, _original=original, **kwargs):
            qlast = q[..., -1:, :].contiguous()
            events.append({'schema': 'trajkv.round9.local_sample.v1', 'purpose': 'calibration',
                'source_id': row['source_id'], 'id': row['id'], 'dataset': row['dataset'],
                'image_sha256': row['original_image_sha256'], 'layer_idx': _index,
                'phase': 'suffix_last' if phase['step'] == 0 else 'teacher_forced_decode', 'step': phase['step'],
                'student_snapshot': {'base_profile': BASE_PROFILE, 'metadata_factor_sha256': BASE_FACTOR_SHA,
                                     'round10_fit_original_student': True},
                'q': qlast.detach().cpu().clone(), 'q_rotated': wht(qlast, half=True).detach().cpu().clone(),
                'exact_keys': kwargs['exact_keys'].detach().cpu().clone(),
                'exact_values': kwargs['exact_values'].detach().cpu().clone(),
                'exact_positions': kwargs['exact_positions'].detach().cpu().clone(),
                'query_positions': kwargs['query_positions'][-1:].detach().cpu().clone(), 'scaling': 128 ** -.5})
            return _original(q, **kwargs)
        owner.attend = wrapped
    selected_tokens = history['token_ids'][:3]
    try:
        with attention_context(model, cache=cache):
            for step, logits in enumerate(feed_teacher(model, batch, cache, selected_tokens)):
                require(bool(torch.isfinite(logits).all()), 'nonfinite fit student logits')
                phase['step'] = step + 1
    finally:
        for owner, original in originals:
            owner.attend = original
    require(len(events) == 28 * len(selected_tokens) and {e['layer_idx'] for e in events} == set(range(28)), 'all-layer fit capture')
    return {'schema': 'trajkv.round10.ls_fit_case.v1', 'source': row, 'layers': layers, 'events': events,
            'visual_positions': batch.visual_positions.cpu(), 'book': owners[0].shared['book'].detach().cpu().clone(),
            'selected_stages': metadata['selected_stages'], 'original_factors': initial_factors,
            'teacher_generated_token_ids': history['token_ids'], 'teacher_cap32': True,
            'fit_capture_scope': 'suffix-last and up to first2 teacher-forced decode; no full-model optimization',
            'all_28_layers_captured': True, 'original_reader_fields': signatures}


def load_fit_samples(items):
    import torch
    from round9_calibration_v1 import validate_sample
    samples = []
    for item in items:
        require(sha(item['file']) == item['sha256'], 'private fit shard changed')
        case = torch.load(item['file'], map_location='cpu', weights_only=True)
        require(case['schema'] == 'trajkv.round10.ls_fit_case.v1' and case['all_28_layers_captured'], 'Round10 fit shard')
        for event in case['events']:
            sample = {**case['layers'][event['layer_idx']], **event, 'book': case['book'],
                      'visual_positions': case['visual_positions'], 'selected_stages': case['selected_stages'],
                      'original_factors': case['original_factors'], '_shard_path': item['file'], '_shard_sha256': item['sha256']}
            validate_sample(sample)
            # The old weighting helper keys only on source_id; namespace raw
            # image names across the four datasets before using that helper.
            sample['original_source_id'] = sample['source_id']
            sample['source_id'] = sample['dataset'] + ':' + sample['source_id']
            samples.append(sample)
    require(len({(s['dataset'], s['source_id']) for s in samples}) == 48 and {s['layer_idx'] for s in samples} == set(range(28)), 'actual fit48/all28 coverage')
    return samples


def replay_branch(model, batch, cache, owners, history):
    import torch
    from attention_bridge import attention_context
    calls = [0]
    originals = []
    for owner in owners:
        old = owner.attend
        originals.append((owner, old))
        def counted(q, *, _old=old, **kwargs):
            calls[0] += 1
            return _old(q, **kwargs)
        owner.attend = counted
    begin = time.perf_counter()
    try:
        values = []
        with attention_context(model, cache=cache):
            for value in feed_teacher(model, batch, cache, history['token_ids']):
                require(bool(torch.isfinite(value).all()), 'nonfinite actual packed TF logits')
                values.append(value.detach().float().cpu().clone())
        require(calls[0] == 28 * len(history['token_ids']), 'actual packed owner not used at every layer/position')
        scores = full_vocab_kl(history['raw_logits'], torch.stack(values))
        scores.update(actual_packed_owner_calls=calls[0], wall_s=time.perf_counter() - begin,
                      question_native_KV_reused=False, question_suffix_recomputed=True,
                      current_query_and_text_KV=True, actual_model_TF_source_evaluation=True)
        return scores
    finally:
        for owner, old in originals:
            owner.attend = old


def technical_one_source(model, processor, row, variant, config, factors, output):
    """Actual prefix/reader controls, not LS fitting or a selection receipt."""
    from ownership import cache_storage
    batch, native, context = prepare_case(model, processor, row)
    history = teacher_history(model, batch, native)
    cache, owners, _, reference = build_cache(native, batch, variant, config)
    cache = owners = None
    theta = {name: factors[name] for name in ('value_head_factors', 'value_center_offsets')}
    results, fb_signature = [], None
    for mode, window in (('LS-FZ', None), ('LS-RE', None), ('FB0', 1), ('FB-W', 1)):
        cache, owners, metadata, signature = build_cache(native, batch, variant, config,
            mode=mode, theta=theta if mode.startswith('LS-') else None, window=window)
        require(signature['K_fields_sha256'] == reference['K_fields_sha256'], 'technical control changed K')
        if mode.startswith('LS-'):
            require(signature['reader_field_sha256'] == reference['reader_field_sha256'], 'technical lambda0 identity')
        elif mode == 'FB0':
            fb_signature = signature['reader_field_sha256']
        else:
            require(signature['reader_field_sha256'] == fb_signature, 'technical W1 must match FB0 fields')
        # Execute all controls even if their fields match, to test true APIs.
        loss = replay_branch(model, batch, cache, owners, history)
        results.append({'mode': mode, 'window': window, 'loss': loss,
            'reader_field_sha256': signature['reader_field_sha256'],
            'K_fields_sha256': signature['K_fields_sha256'], 'metadata': metadata,
            'actual_owner': cache_storage(cache, cache.visual_positions)})
        cache = owners = None
        gc.collect()
    result = {'schema': 'trajkv.round10.calibration_technical.v1',
        'status': 'PASS_TECHNICAL_ONE_SOURCE_LAMBDA0_W1_ACTUAL_MODEL',
        'actual_GPU': True, 'context': context, 'controls': results,
        'teacher_token_ids': history['token_ids'], 'teacher_sources': 1,
        'teacher_tokens': len(history['token_ids']), 'actual_TF_source_evaluations': 4,
        'full_fit48_select16_completed': False, 'theta_selected': None,
        'window_selected': None, 'references_read': False,
        'new_free_generation_task_answers': 0, 'quality_claim': False}
    return result


def calibrate(config, output):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from round10_inputs_v1 import load_round10_inputs
    from r7_frozen_pilot_v2 import frozen_variant
    from round9_calibration_v1 import value_ls_warmstart, make_legacy_callbacks
    from attention_bridge import language_attention_modules
    from ownership import cache_storage
    from model_registry import get_model_spec
    from evaluate import model_identity
    validate_config(config)
    source_before = check_sources(config)
    rows = {}
    for role in ('fit', 'select'):
        rows[role] = load_round10_inputs(config[f'cal_{role}_inputs'], config.get('input_manifest_path'),
            expected_sha256=config[f'cal_{role}_sha256'], expected_role=config.get(f'cal_{role}_role', f'cal_{role}'))
    validate_rows(rows['fit'], rows['select'])
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6), 'actual CUDA SM86 required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variant = frozen_variant(config)
    require(sha(variant['metadata_factor_path']) == BASE_FACTOR_SHA, 'original V0 factor identity')
    factors = json.loads(Path(variant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    identity = model_identity(config, get_model_spec('M1'))
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
        device_map={'': 0}, attn_implementation='sdpa', local_files_only=True).eval()
    model.requires_grad_(False)
    require(all(p.dtype == torch.float16 for p in model.parameters()), 'native FP16 unquantized weights')
    require(int(model.generation_config.num_beams or 1) == 1, 'native greedy beam-one')
    _, modules = language_attention_modules(model)
    require(len(modules) == 28, 'M1 all28 geometry')
    manifest = {'schema': 'trajkv.round10.calibration_run.v1', 'config': config, 'source_hashes': source_before,
        'model_identity': identity, 'expected_fit_sources': 48, 'expected_select_sources': 16,
        'teacher_generation': {'max_new_tokens': 32, 'native_generation_config': model.generation_config.to_dict()},
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'python_version': platform.python_version(), 'actual_GPU': torch.cuda.get_device_name(),
        'weight_dtype': 'float16', 'tf32': False, 'references_read': False,
        'private_native_prefix_and_WO_not_public': True, 'deployment_peak_or_speed_benchmark': False,
        'question_unknown_encoding': True, 'fit_capture_scope': 'suffix-last plus up to first2 teacher-forced states/all28; not full-suffix fitting',
        'actual_processor_class': type(processor).__module__ + '.' + type(processor).__name__,
        'actual_tokenizer_class': type(processor.tokenizer).__module__ + '.' + type(processor.tokenizer).__name__,
        'actual_tokenizer_is_fast': bool(getattr(processor.tokenizer, 'is_fast', False)),
        'teacher_loss_arithmetic': 'stable CPU float64 from actual GPU raw logits'}
    write_new(output / 'manifest.json', manifest)
    if config.get('technical_only', False):
        index = config.get('technical_source_index', 0)
        require(type(index) is int and 0 <= index < 16, 'technical source index0..15')
        result = technical_one_source(model, processor, rows['select'][index], variant, config, factors, output)
        require(check_sources(config) == source_before, 'source changed during technical run')
        result['source_hashes'] = source_before
        write_new(output / 'complete.json', result)
        return result
    records = (output / 'records.jsonl').open('x', encoding='utf-8')
    def log(row):
        records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        records.flush()
        print(json.dumps({k: row[k] for k in ('phase', 'index', 'id') if k in row}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    fit_items = []
    counts = Counter()
    try:
        for index, row in enumerate(rows['fit']):
            begin = time.perf_counter()
            batch, native, context = prepare_case(model, processor, row)
            history = teacher_history(model, batch, native)
            counts['teacher_generated_sources'] += 1
            counts['teacher_generated_tokens'] += len(history['token_ids'])
            case = capture_fit_case(model, batch, native, row, variant, config, history, factors)
            path = output / 'private_fit_shards' / f'{index:03d}.pt'
            path.parent.mkdir(exist_ok=True)
            torch.save(case, path)
            item = {'index': index, 'id': row['id'], 'dataset': row['dataset'], 'source_id': row['source_id'],
                    'file': str(path), 'sha256': sha(path), 'events': len(case['events']),
                    'bytes': path.stat().st_size, 'context': context}
            fit_items.append(item)
            counts['fit_student_outer_forwards'] += len(case['events']) // 28
            log({'phase': 'fit_source_captured', **item, 'wall_s': time.perf_counter() - begin})
            case = history = batch = native = None
            gc.collect()
            torch.cuda.empty_cache()
        wo_file = output / 'wo_private.pt'
        torch.save({i: module.o_proj.weight.detach().cpu().clone() for i, module in enumerate(modules)}, wo_file)
        write_new(output / 'fit_collection_complete.json', {'schema': 'trajkv.round10.fit_collection.v1',
            'status': 'PASS_REAL_MODEL_FIT48_ALL28', 'sources': fit_items, 'private_WO_file': str(wo_file),
            'private_WO_sha256': sha(wo_file), 'references_read': False, 'actual_GPU': True})
        samples = load_fit_samples(fit_items)
        callbacks = make_legacy_callbacks(config['legacy_a2'], config['legacy_experiments'], config['legacy_scripts'])
        cfg = {**callbacks, 'device': 'cuda:0', 'wo_by_layer': {i: m.o_proj.weight.detach() for i, m in enumerate(modules)},
               'teacher_reader': local_teacher, 'gain_floor': config.get('gain_floor', 1e-4),
               'ls_ridge': config.get('ls_ridge', 1e-5)}
        g0 = torch.as_tensor(factors['value_head_factors'], dtype=torch.float32)
        o0 = torch.as_tensor(factors['value_center_offsets'], dtype=torch.float32)
        proposal_g, proposal_h = value_ls_warmstart(samples, g0, g0 * o0, cfg)
        provenance = {'module_sha256': sha(__file__), 'source_hashes': source_before,
            'fit_inputs_sha256': config['cal_fit_sha256'], 'fit_sources': 48,
            'fit_shards': [{k: item[k] for k in ('file', 'sha256', 'source_id', 'dataset')} for item in fit_items],
            'private_WO_sha256': sha(wo_file), 'conditional_LS_only': True, 'training_updates': 0,
            'fixed_original_student_q_text_and_discrete_fields': True,
            'whole_model_gradient_or_RE_aware_fit_claimed': False, 'references_read': False}
        proposal = theta_document(interpolate_theta(factors, proposal_g, proposal_h, 1.), provenance)
        write_new(output / 'value_theta_LS_proposal.json', proposal)
        theta_grid = {coefficient: interpolate_theta(factors, proposal_g, proposal_h, coefficient) for coefficient in LAMBDAS}
        samples = None
        gc.collect()
        expected_sources = [r['dataset'] + ':' + r['source_id'] for r in rows['select']]
        losses = {x: {'LS-FZ': {}, 'LS-RE': {}} for x in LAMBDAS}
        window_losses = {w: {} for w in WINDOWS}
        for index, row in enumerate(rows['select']):
            batch, native, context = prepare_case(model, processor, row)
            history = teacher_history(model, batch, native)
            counts['teacher_generated_sources'] += 1
            counts['teacher_generated_tokens'] += len(history['token_ids'])
            source = row['dataset'] + ':' + row['source_id']
            teacher_file = output / 'private_select_teachers' / f'{index:03d}.pt'
            teacher_file.parent.mkdir(exist_ok=True)
            torch.save({'schema': 'trajkv.round10.select_teacher.v1', 'source': row,
                        'context': context, **history}, teacher_file)
            log({'phase': 'select_source_teacher', 'index': index, 'id': row['id'],
                 'dataset': row['dataset'], 'source_id': row['source_id'],
                 'context': context, 'teacher_token_ids': history['token_ids'],
                 'first_call': history['first_call'], 'teacher_logits_sha256': tensor_sha(history['raw_logits']),
                 'teacher_file': str(teacher_file), 'teacher_file_sha256': sha(teacher_file),
                 'EOS_reached': history['EOS_reached'], 'cap': 32, 'references_read': False})
            original_cache, original_owners, original_metadata, original_signature = build_cache(native, batch, variant, config)
            original_cache = original_owners = original_metadata = None
            dedup = {}
            branches = [(f'lambda{coefficient}_{mode}', mode, coefficient, None)
                        for coefficient in LAMBDAS for mode in ('LS-FZ', 'LS-RE')]
            branches += [(f'window{window}', 'FB0' if window == 1 else 'FB-W', None, window) for window in WINDOWS]
            for label, mode, coefficient, window in branches:
                cache, owners, metadata, signature = build_cache(native, batch, variant, config, mode=mode,
                    theta=theta_grid[coefficient] if coefficient is not None else None, window=window)
                require(signature['K_fields_sha256'] == original_signature['K_fields_sha256'], 'candidate changed original K code/gain/mean/residuals')
                if coefficient == 0:
                    require(signature['reader_field_sha256'] == original_signature['reader_field_sha256'], 'lambda0 failed actual reader-field identity')
                owner = cache_storage(cache, cache.visual_positions)
                key = signature['reader_field_sha256']
                if key in dedup:
                    scores = dict(dedup[key][1])
                    reused = dedup[key][0]
                    scores['reused_producer_packed_owner_calls'] = scores['actual_packed_owner_calls']
                    scores['actual_packed_owner_calls'] = 0
                    scores['actual_model_TF_source_evaluation'] = False
                    scores['exact_fields_TF_result_reused'] = True
                    counts['TF_source_evaluations_reused_exact_fields'] += 1
                else:
                    scores = replay_branch(model, batch, cache, owners, history)
                    dedup[key] = (label, scores)
                    reused = None
                    counts['TF_source_evaluations_actual'] += 1
                    counts['TF_student_outer_forwards_actual'] += scores['positions']
                entry = {'phase': 'select_branch', 'index': index, 'id': row['id'], 'dataset': row['dataset'],
                    'source_id': row['source_id'], 'context': context, 'branch': label, 'mode': mode,
                    'lambda': coefficient, 'window': window, 'loss': scores,
                    'reader_field_sha256': key, 'K_fields_sha256': signature['K_fields_sha256'],
                    'same_case_exact_field_reuse_from': reused, 'owner_before_question_suffix': owner,
                    'owner_metadata': metadata, 'calibration_keeps_native_reference': True,
                    'not_deployment_peak_or_quality_measurement': True}
                log(entry)
                if coefficient is not None:
                    losses[coefficient][mode][source] = scores['source_mean_KL']
                else:
                    window_losses[window][source] = scores['source_mean_KL']
                cache = owners = metadata = signature = None
                gc.collect()
            batch = native = history = original_signature = None
            torch.cuda.empty_cache()
        coefficient, lambda_summary = select_common_lambda(losses, expected_sources)
        window, window_summary = select_window(window_losses, expected_sources)
        provenance.update(select_inputs_sha256=config['cal_select_sha256'], select_sources=16)
        provenance.update({'selected_common_lambda': coefficient, 'lambda_grid': list(LAMBDAS),
            'identity_is_control_not_candidate_gate': True, 'nonzero_proposal_research_test_not_quality_certification': True,
            'selection_objective': 'source-balanced mean of FZ and RE full-vocabulary actual-model TF KL',
            'select_teacher_EOS_cap32': True})
        selected_doc = theta_document(theta_grid[coefficient], provenance)
        write_new(output / 'value_theta_selected.json', selected_doc)
        write_new(output / 'window_selected.json', {'schema': 'trajkv.round10.window_selection.v1',
            'window': window, 'window_grid': list(WINDOWS), 'tile_tokens': 256,
            'global_resets_at_tile': False, 'finite_windows_reset_at_tile': True,
            'W1_is_control_not_candidate_gate': True, 'selected_on_calSelect_only': True,
            'cal_select_inputs_sha256': config['cal_select_sha256'],
            'selection_objective': 'source-balanced full-vocabulary actual-model TF KL',
            'source_hashes': source_before, 'free_generation_quality_claim': False})
        require(check_sources(config) == source_before, 'source changed during run')
        require(all(sha(item['file']) == item['sha256'] for item in fit_items), 'fit tensor changed')
        result = {'schema': 'trajkv.round10.calibration_complete.v1', 'status': 'PASS_LS48_AND_ACTUAL_MODEL_SELECT16',
            'actual_GPU': True, 'fit_sources': 48, 'select_sources': 16, 'teacher_cap': 32,
            'selected_common_lambda': coefficient, 'selected_window': window,
            'lambda_scores': {str(k): v for k, v in lambda_summary.items()},
            'window_scores': {str(k): v for k, v in window_summary.items()}, 'counts': dict(counts),
            'theta_file': str(output / 'value_theta_selected.json'), 'theta_sha256': sha(output / 'value_theta_selected.json'),
            'window_file': str(output / 'window_selected.json'), 'window_sha256': sha(output / 'window_selected.json'),
            'source_hashes': source_before, 'private_WO_sha256': sha(wo_file),
            'records_sha256': sha(output / 'records.jsonl'), 'references_read': False,
            'quality_or_improved_incumbent_claim': False, 'new_free_generation_task_answers': 0,
            'teacher_generation_is_separately_counted': True, 'wall_s': time.perf_counter() - started}
        write_new(output / 'complete.json', result)
        return result
    except Exception as error:
        write_new(output / 'failure.json', {'status': 'FAILED_ACTUAL_CALIBRATION_NO_SYNTHETIC_COMPLETION',
            'error_type': type(error).__name__, 'error': str(error), 'completed_fit_sources': len(fit_items),
            'counts': dict(counts), 'references_read': False, 'new_task_quality_claim': False})
        raise
    finally:
        records.close()


def self_check(output):
    import torch
    from types import SimpleNamespace
    from round9_calibration_v1 import solve_positive_gain_ls
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    def passed(name): checks.append(name)
    base = {'value_head_factors': torch.ones(28, 4), 'value_center_offsets': torch.full((28, 4), .13)}
    pg, ph = torch.full((28, 4), 2.), torch.full((28, 4), .6)
    zero = interpolate_theta(base, pg, ph, 0.)
    require(torch.equal(zero['value_center_offsets'], base['value_center_offsets']), 'lambda0 offsets bitwise')
    passed('absolute_theta0_preserves_original_offsets_without_roundtrip')
    half = interpolate_theta(base, pg, ph, .5)
    require(torch.equal(half['value_head_factors'], torch.full((28, 4), 1.5)) and
            torch.allclose(half['value_center_offsets'] * 1.5, torch.full((28, 4), .365)), 'interpolate g/h')
    passed('interpolation_in_g_h_not_gain_offset')
    logits = torch.tensor([[10000., 9999., -10000.], [-3., 2., 1.]])
    require(abs(full_vocab_kl(logits, logits)['source_mean_KL']) < 1e-14, 'stable KL identity')
    require(full_vocab_kl(logits, logits + torch.tensor([0., 1., 0.]))['source_mean_KL'] > 0, 'stable KL actual difference')
    passed('full_vocab_stable_CPU_F64_identity_and_positive_difference')
    H = torch.eye(8)
    unconstrained = torch.tensor([-2., 2., 3., 4., 0., 1., 2., 3.])
    solved = solve_positive_gain_ls(H, unconstrained, torch.ones(8), gain_floor=.1, ridge=1e-9)
    require(abs(float(solved[0]) - .1) < 1e-5 and bool((solved[:4] >= .1).all()), 'active positive constraint')
    passed('reuse_actual_positive_gain_active_set_solver')
    sources = ['s0', 's1']
    losses = {x: {'LS-FZ': {s: 0. if x == 0 else x for s in sources},
                  'LS-RE': {s: 0. if x == 0 else 1. / x for s in sources}} for x in LAMBDAS}
    selected, _ = select_common_lambda(losses, sources)
    require(selected == 1., 'must choose common average, exclude zero control')
    passed('common_lambda_not_independent_branch_winners_or_identity_gate')
    windows = {w: {s: 0. if w == 1 else 1. for s in sources} for w in WINDOWS}
    chosen, _ = select_window(windows, sources)
    require(chosen == 32, 'nontrivial window selection/tie')
    passed('W1_remains_control_shorter_nontrivial_tie_wins')
    broken = {x: {m: dict(v) for m, v in branches.items()} for x, branches in losses.items()}
    broken[.25]['LS-FZ'].pop('s1')
    try: select_common_lambda(broken, sources)
    except ValueError: passed('missing_source_rejected_not_success_subset')
    else: raise AssertionError('missing source accepted')
    try: full_vocab_kl(logits, logits[:1])
    except ValueError: passed('teacher_history_position_mismatch_rejected')
    else: raise AssertionError('shape mismatch accepted')
    class FakeCache:
        def __init__(self, length): self.length = length
        def get_seq_length(self): return self.length
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generation_config = SimpleNamespace(eos_token_id=3)
            self.actual_inputs = []
        def forward(self, input_ids, attention_mask, position_ids, past_key_values,
                    use_cache=True, return_dict=True, logits_to_keep=1):
            start = past_key_values.length
            require(torch.equal(position_ids, torch.arange(start, start + input_ids.shape[-1])[None]),
                    'CPU facade actual position identity')
            past_key_values.length += input_ids.shape[-1]
            require(attention_mask.shape[-1] == past_key_values.length, 'CPU facade actual mask length')
            self.actual_inputs.append(input_ids.clone())
            return SimpleNamespace(logits=torch.tensor([[[1., 2., 0., -1.]]]))
    fake = FakeModel()
    batch = SimpleNamespace(ids=torch.tensor([[10, 11, 12, 13, 14]]),
        mask=torch.ones(1, 5, dtype=torch.long), positions=torch.arange(5)[None],
        boundary=3, full_length=5)
    cache = FakeCache(3)
    values = list(feed_teacher(fake, batch, cache, [1, 2]))
    require(len(values) == 2 and cache.length == 6 and
            torch.equal(fake.actual_inputs[0], torch.tensor([[13, 14]])) and
            torch.equal(fake.actual_inputs[1], torch.tensor([[1]])), 'fresh suffix and teacher history replay')
    passed('real_frozen_suffix_helper_CPU_facade_recomputes_question_then_teacher_token')
    try: list(feed_teacher(fake, batch, FakeCache(5), [1]))
    except ValueError: passed('already_written_question_cache_rejected_by_actual_forward_hook')
    else: raise AssertionError('native question KV accepted')
    state = SimpleNamespace(direction_code=torch.zeros((1, 4, 32, 16), dtype=torch.uint8),
        direction_gain=torch.ones((1, 4, 32, 1), dtype=torch.float16),
        direction_mean=torch.zeros((1, 4, 1, 128), dtype=torch.float16),
        v_code=torch.zeros((1, 4, 128, 1), dtype=torch.int32),
        v_scale=torch.ones((1, 4, 128, 1), dtype=torch.float16),
        v_min=torch.zeros((1, 4, 128, 1), dtype=torch.float16),
        depth_residual_lookup=torch.full((2, 4, 2), -1, dtype=torch.int32))
    fake_owner = SimpleNamespace(state=state)
    old_sig = state_signatures([fake_owner])
    state.v_code[0, 0, 0, 0] = 1
    new_sig = state_signatures([fake_owner])
    require(old_sig['reader_field_sha256'] != new_sig['reader_field_sha256'] and
            old_sig['K_fields_sha256'] == new_sig['K_fields_sha256'], 'empty-pool state/K identity')
    passed('empty_residual_layer_signature_and_V_only_change_keep_K_signature')
    result = {'status': 'PASS_CPU_ARITHMETIC_AND_SELECTION_CONTROLS', 'module_sha256': sha(__file__),
              'checks': checks, 'actual_GPU': False, 'actual_model_or_factory_executed': False,
              'free_generation_quality_claim': False}
    write_new(output / 'CPU_RECEIPT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        result = self_check(args.output)
    else:
        require(args.config is not None, '--config required for actual calibration')
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        config['calibration_config_sha256'] = sha(args.config)
        result = calibrate(config, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
