"""Conditional fixed-q Base/Residual exchange selection, stable F64 version2.

No model, task scorer, answers, transport, or automatic GPU work. The real CLI
uses the original compressed-student suffix-last q/text snapshots. Full joint
visual+text attention KL is the only profile-selection objective. Projection
diagnostics retain V0 and are reported separately, never used as extra losses.
Independent fork of immutable v1 after an actual Torch2.4.1/cu121/SM86 F64
log_softmax normalization failure. No KL threshold has been relaxed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import torch
import round9_calibration_v1 as cal

CAL_SHA = '757cc51740ad2f92d8ebd9ad9adadf2719131ae634b16b24052363f8e45ba84f'
COLLECTION_SHA = 'c838b3a79cb970233a9579f8972ae4276157fad1001fd9b4e9118945bc0dc521'
COLLECTOR_SHA = '2d709806122c8127a766747ef5d29821fb22f389238f8448f0dfd25c363a61df'
LOW_BOOK_SHA = 'af29f26ddcbdf4ac6f098c3df497d2d741f0e7909c13ecc090194446846c9589'
HIGH_BOOK_SHA = '9d3ad3683cf6888c30f3c8ba9c6dd0d4dbcf40106a34567d578479245e7309ae'
FACTOR_SHA = '8f47c56af5fe5e4da99bae05b83084173c5a55601c46c91130894075907e9cea'
PHASE = 'suffix_last'
SCHEMA = 'trajkv.round9.conditional_precision_exchange_calibration.v2'
PARENT_V1_SHA = '1ce11daff0a1b74e22f54f4dd9dfb5308d541a49e0532fa2cc1ae7e0bd8a6f6a'
ROW_SUM_ATOL = 1e-12
CPU_GPU_LOG_ATOL = 1e-10


def require(value, message):
    if not value:
        raise ValueError('ROUND9_EXCHANGE_CAL: ' + message)


def sha(path):
    return cal.sha(path)


def canonical(value):
    return cal.canonical(value)


def write(path, value):
    cal.write_new(path, value)


def repeat(value):
    return value.repeat_interleave(7, 1)


def stable_logsumexp(value, dim=-1, keepdim=False):
    """Explicit amax/shift/exp/sum/log; permits masked negative infinity only."""
    require(value.dtype == torch.float64, 'stable probability arithmetic must be FP64')
    require(bool((torch.isfinite(value) | torch.isneginf(value)).all()), 'NaN/+Inf in log normalization input')
    maximum = value.amax(dim=dim, keepdim=True)
    require(bool(torch.isfinite(maximum).all()), 'all-masked log normalization row')
    shifted = value - maximum
    partition = shifted.exp().sum(dim=dim, keepdim=True)
    require(bool(torch.isfinite(partition).all() and (partition > 0).all()), 'invalid positive partition')
    result = maximum + partition.log()
    return result if keepdim else result.squeeze(dim)


def check_log_distribution(logp, dim=-1):
    require(logp.dtype == torch.float64 and
            bool((torch.isfinite(logp) | torch.isneginf(logp)).all()), 'invalid FP64 log probabilities')
    probability = logp.exp()
    require(bool(torch.isfinite(probability).all()), 'nonfinite attention probability')
    row_error = float((probability.sum(dim=dim) - 1).abs().max())
    require(row_error <= ROW_SUM_ATOL, 'probability row is not normalized: max_error=' + repr(row_error))
    return row_error


def stable_log_softmax(value, dim=-1):
    """Shift-first form avoids the known failing fused FP64 CUDA operation."""
    require(value.dtype == torch.float64, 'FP64 log-softmax input required')
    require(bool((torch.isfinite(value) | torch.isneginf(value)).all()), 'NaN/+Inf logits')
    maximum = value.amax(dim=dim, keepdim=True)
    require(bool(torch.isfinite(maximum).all()), 'all-masked attention row')
    shifted = value - maximum
    partition = shifted.exp().sum(dim=dim, keepdim=True)
    require(bool(torch.isfinite(partition).all() and (partition > 0).all()), 'invalid attention partition')
    logp = shifted - partition.log()
    check_log_distribution(logp, dim)
    return logp


def mask_k_units(fields, dropped, layer):
    """Private calibration view only: old pools remain; no byte saving claim."""
    result = dict(fields)
    lookup = fields['depth_residual_lookup'].clone()
    for item in dropped:
        require(len(item) == 3, 'K donors are [layer,head,stage] triples')
        l, h, stage = map(int, item)
        require(0 <= l < 28 and 0 <= h < 4 and stage == 0, 'original stage0 K unit required')
        if l == layer:
            require(int(lookup[0, h, stage]) >= 0 and int(lookup[0, h, 1]) < 0,
                    'donor absent, or a child would remain')
            lookup[0, h, stage] = -1
    result['depth_residual_lookup'] = lookup
    require(torch.equal(lookup[1], fields['depth_residual_lookup'][1]), 'V lookup changed')
    return result


def joint_distribution(s, fields=None, book=None, *, native=False, decode_key=None):
    """FP64 fixed-q logits, including original exact-text scores and causal mask."""
    q = s['q'].double()
    scale = float(s['scaling'])
    if native:
        visual = q @ repeat(s['visual_k']).double().transpose(-1, -2) * scale
    else:
        require(callable(decode_key), 'actual sequential-Half 16/256 decoder required')
        key = decode_key(fields, book)
        visual = s['q_rotated'].double() @ repeat(key).double().transpose(-1, -2) * scale
        visual += (q * repeat(fields['direction_mean']).double()).sum(-1, keepdim=True) * scale
    text = q @ repeat(s['exact_keys']).double().transpose(-1, -2) * scale
    positions = torch.cat((s['visual_positions'], s['exact_positions']))
    visible = positions[None, :] <= s['query_positions'][:, None]
    require(bool(visible.any(-1).all()), 'empty causal support')
    logits = torch.cat((visual, text), -1).masked_fill(~visible[None, None], -torch.inf)
    logp = stable_log_softmax(logits, -1)
    probability = logp.exp()
    require(bool(torch.isfinite(probability).all()), 'nonfinite attention probabilities')
    return {'logp': logp, 'p': probability, 'visible': visible,
            'visual_logits': visual, 'visual_n': int(s['visual_positions'].numel()),
            'joint_logits': logits}


def joint_kl_by_head(native, candidate):
    require(torch.equal(native['visible'], candidate['visible']), 'attention support differs')
    term = native['p'] * (native['logp'] - candidate['logp'])
    result = torch.where(native['visible'][None, None], term, torch.zeros_like(term)).sum(-1)
    require(bool(torch.isfinite(result).all()) and float(result.min()) >= -1e-10,
            'invalid full attention KL')
    return result


def route_metrics(native, candidate):
    n = native['visual_n']
    lf, lc = native['logp'], candidate['logp']
    check_log_distribution(lf)
    check_log_distribution(lc)
    lmf, lmc = stable_logsumexp(lf[..., :n], -1), stable_logsumexp(lc[..., :n], -1)
    ltf, ltc = stable_logsumexp(lf[..., n:], -1), stable_logsumexp(lc[..., n:], -1)
    mf, tf = lmf.exp(), ltf.exp()
    require(bool(torch.isfinite(ltf).all()), 'at least one exact text key must be visible')
    mass = mf * (lmf - lmc) + tf * (ltf - ltc)
    vf = stable_log_softmax(native['visual_logits'], -1)
    vc = stable_log_softmax(candidate['visual_logits'], -1)
    conditional = (vf.exp() * (vf - vc)).sum(-1)
    visible_text = native['visible'][..., n:][None, None]
    text_term = torch.where(visible_text,
        lf[..., n:].exp() * (lf[..., n:] - ltf[..., None] - lc[..., n:] + ltc[..., None]),
        torch.zeros_like(lf[..., n:])).sum(-1)
    kl = joint_kl_by_head(native, candidate)
    residual = float((kl - mass - mf * conditional - text_term).abs().max())
    require(residual <= 1e-8, 'full KL mass/routing/text chain failed')
    return {'joint_attention_KL': float(kl.mean()), 'modality_mass_KL': float(mass.mean()),
            'native_mass_weighted_visual_routing_KL': float((mf * conditional).mean()),
            'conditional_visual_KL': float(conditional.mean()),
            'weighted_conditional_text_KL': float(text_term.mean()),
            'native_visual_mass': float(mf.mean()), 'candidate_visual_mass': float(lmc.exp().mean()),
            'visual_routing_TV': float((.5 * (vf.exp() - vc.exp()).abs().sum(-1)).mean()),
            'visual_top1_changed_fraction': float((vf.argmax(-1) != vc.argmax(-1)).double().mean()),
            'KL_chain_max_abs': residual}


def projected_arms(s, native, candidate, value0, wo):
    """Dense FP64 attention -> Half head outputs -> frozen Half W_O projection."""
    n = native['visual_n']
    true_v, text_v = repeat(s['visual_v']).double(), repeat(s['exact_values']).double()
    value0 = repeat(value0).double()
    arms = {}
    for name, dist, values in [('FF', native, true_v), ('CF', candidate, true_v),
                               ('FC', native, value0), ('CC', candidate, value0)]:
        heads = (dist['p'][..., :n] @ values + dist['p'][..., n:] @ text_v).half()
        arms[name] = cal.project(heads, wo).float()
    require(all(bool(torch.isfinite(x).all()) for x in arms.values()), 'nonfinite local projection')
    mse = lambda x: float(x.square().mean())
    losses = {name: mse(value - arms['FF']) for name, value in arms.items()}
    interaction = arms['CC'] - arms['CF'] - arms['FC'] + arms['FF']
    return {'MSE_to_trueK_trueV': losses,
            'Ktrue_target_keeping_V0_MSE': mse(arms['CC'] - arms['FC']),
            'K_component_when_V_true_MSE': losses['CF'],
            'V_component_when_K_true_MSE': losses['FC'],
            'K_V_interaction_vector_MSE': mse(interaction),
            'projected_loss_Sh_K': .5 * (losses['CF'] + losses['CC'] - losses['FC']),
            'projected_loss_Sh_V': .5 * (losses['FC'] + losses['CC'] - losses['CF']),
            'arm_definition': 'K first, V second; F=native visual, C=actual packed reconstruction; same student q/text'}


def select_profile(base_loss, upgrades, drop_old, drop_upgraded):
    """Exact conditional minimum for head-separable fixed-q KL, not task loss."""
    units = sorted(drop_old)
    require(len(units) >= 4 and all(len(x) == 3 and x[2] == 0 for x in units), 'K stage0 donor set')
    rows = []
    for layer in sorted(upgrades):
        conditional = {u: drop_upgraded[layer][u] if u[0] == layer else drop_old[u] for u in units}
        require(all(math.isfinite(v) for v in conditional.values()), 'nonfinite donor marginal')
        chosen = sorted(units, key=lambda u: (conditional[u], u))[:4]
        loss = base_loss + upgrades[layer] + sum(conditional[u] for u in chosen)
        rows.append({'upgraded_layer': layer, 'upgrade_keep_all_K_residual_route_delta': upgrades[layer],
            'dropped_k_units': [list(u) for u in chosen],
            'conditional_donor_costs': [{'unit': list(u), 'cost': conditional[u],
                'evaluated_under_new_layer_base_and_RE': u[0] == layer, 'chosen': u in chosen}
                for u in sorted(units, key=lambda u: (conditional[u], u))],
            'route_loss_predicted': loss})
    require(rows, 'no upgrade layer candidates')
    best = min(rows, key=lambda x: (x['route_loss_predicted'], x['upgraded_layer'], x['dropped_k_units']))
    return best, rows


def phase_samples(samples):
    selected = [s for s in samples if s['phase'] == PHASE and s['step'] == 0]
    groups = defaultdict(list)
    for s in selected:
        require(s['q'].shape == (1, 28, 1, 128), 'last-prefill query must be exactly one token')
        groups[s['source_id']].append(s)
    require(len(groups) == 32, 'exact32 calibration sources')
    for rows in groups.values():
        require(len(rows) == 28 and {s['layer_idx'] for s in rows} == set(range(28)), 'one event per28 layers/source')
    require(Counter(rows[0]['dataset'] for rows in groups.values()) == {'docvqa': 8, 'textvqa': 16, 'infovqa': 8},
            'original Doc8/Text16/Info8 calibration mixture required')
    original = selected[0]
    stages = original['selected_stages']
    require(len(stages) == 43 and all(x[3] == 0 for x in stages), 'fixed43 stage0 coverage')
    require(sum(x[0] == 'k' for x in stages) == 23 and sum(x[0] == 'v' for x in stages) == 20,
            'original23 K/20 V residuals')
    for s in selected:
        snap = s['student_snapshot']
        require(snap.get('collector_source_sha256') == COLLECTOR_SHA and
                snap.get('direction_book_path') == LOW_BOOK_SHA and snap.get('metadata_factor_path') == FACTOR_SHA and
                snap.get('student_theta_sha256') is None, 'original compressed-student snapshot only')
        require(canonical(s['selected_stages']) == canonical(stages) and
                s['original_factors'] == original['original_factors'] and torch.equal(s['book'], original['book']),
                'mixed original coverage/factors/book')
        require('visual_k_rotated' in s and s['visual_k_rotated'].dtype == torch.float32,
                'actual original centered-WHT FP32 source required')
        require('native_q' not in s and 'native_suffix_queries' not in s, 'this profile uses original compressed q only')
    return sorted(selected, key=lambda s: (s['source_id'], s['layer_idx'])), stages


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['layer_idx'])].append(row)
    output = []
    for (dataset, layer), items in sorted(groups.items()):
        route_names = items[0]['routing'].keys()
        proj_names = [k for k, v in items[0]['projection'].items() if isinstance(v, (float, int))]
        output.append({'dataset': dataset, 'layer': layer, 'sources': len(items),
            'routing': {k: sum(x['routing'][k] for x in items) / len(items) for k in route_names},
            'projection': {k: sum(x['projection'][k] for x in items) / len(items) for k in proj_names},
            'four_arm_MSE_to_FF': {k: sum(x['projection']['MSE_to_trueK_trueV'][k] for x in items) / len(items)
                for k in ('FF', 'CF', 'FC', 'CC')}})
    return output


@torch.no_grad()
def calibrate(config, output):
    started = time.perf_counter()
    require(sha(cal.__file__) == CAL_SHA, 'frozen cal decoder/loader source differs')
    output = Path(output)
    require(not output.exists(), 'immutable analysis output already exists')
    directory = Path(config['collection_dir'])
    if not (directory / 'complete.json').is_file():
        return {'status': 'WAIT_CLOSED_ORIGINAL_CAL32', 'profile_selected': False, 'analysis_written': False,
                'new_model_forwards': 0}
    require(config.get('collection_complete_sha256') == COLLECTION_SHA and sha(directory / 'complete.json') == COLLECTION_SHA,
            'exact original actual cal32 complete source required')
    require(config.get('phase', PHASE) == PHASE, 'first prototype is suffix_last only')
    require(config.get('high_book_sha256') == HIGH_BOOK_SHA and sha(config['high_book_path']) == HIGH_BOOK_SHA,
            'original frozen Gaussian G4_256 source required')
    ops = importlib.import_module('round9_precision_exchange_v1')
    require(config.get('exchange_helper_sha256') == sha(ops.__file__), 'exact reviewed exchange helper SHA required')
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6), 'real SM86 CUDA required')
    device = config.get('device', 'cuda:0')
    require(str(device).startswith('cuda'), 'actual GPU calibration, not a CPU proxy')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    callbacks = cal.make_legacy_callbacks(config['legacy_a2'], config['legacy_experiments'], config['legacy_scripts'])
    high_data = json.loads(Path(config['high_book_path']).read_text(encoding='utf-8'))
    high_book = torch.tensor(high_data['centers'], dtype=torch.float16, device=device)
    require(high_book.shape == (256, 4) and bool(torch.isfinite(high_book).all()), 'high book geometry')
    all_samples, wos, collection_proof = cal.load_collection(directory)
    samples, stages = phase_samples(all_samples)
    units = sorted((l, h, depth) for axis, l, h, depth in stages if axis == 'k')
    weight = 1 / (32 * 28)
    source_before = {'calibrator': sha(__file__), 'cal_helper': sha(cal.__file__),
        'exchange_helper': sha(ops.__file__), 'high_book': sha(config['high_book_path']),
        'collection_complete': sha(directory / 'complete.json'),
        'collection_manifest': sha(directory / 'manifest.json'),
        'configuration_canonical_sha256': canonical(config), 'callback_sources': callbacks['legacy_source_hashes']}
    projection_cfg = {'wo_by_layer': wos}
    base_loss = 0.
    upgrades = {l: 0. for l in range(28)}
    old_drops = {u: 0. for u in units}
    new_drops = {l: {u: 0. for u in units if u[0] == l} for l in range(28)}
    baseline_records = []
    for index, raw in enumerate(samples):
        s = cal.to_device(raw, device)
        layer = s['layer_idx']
        factors = torch.tensor(s['original_factors']['key_head_factors'][layer], dtype=torch.float32, device=device)
        fields, book = s['r7_fields'], s['book']
        native = joint_distribution(s, native=True)
        before = joint_distribution(s, fields, book, decode_key=ops.decode_exchange_key_fields)
        before_kl = float(joint_kl_by_head(native, before).mean())
        base_loss += weight * before_kl
        value0 = cal.decoded_value(fields)
        wo = cal._wo(projection_cfg, layer, device)
        baseline_records.append({'source_id': s['source_id'], 'id': s['id'], 'dataset': s['dataset'],
            'layer_idx': layer, 'phase': PHASE, 'step': 0,
            'routing': route_metrics(native, before), 'projection': projected_arms(s, native, before, value0, wo)})
        upgraded, upper_book, _ = ops.precision_exchange_fields(fields, s['visual_k_rotated'], book, high_book, factors,
            layer_idx=layer, upgraded_layer=layer, dropped_k_units=[], upgrade=True, compact=False,
            encode_key=callbacks['encode_key'], decode_key=callbacks['decode_key'], encode_sign_depth=callbacks['encode_sign_depth'])
        upper = joint_distribution(s, upgraded, upper_book, decode_key=ops.decode_exchange_key_fields)
        upper_kl = float(joint_kl_by_head(native, upper).mean())
        upgrades[layer] += weight * (upper_kl - before_kl)
        for unit in (u for u in units if u[0] == layer):
            dropped = joint_distribution(s, mask_k_units(fields, [unit], layer), book,
                                         decode_key=ops.decode_exchange_key_fields)
            old_drops[unit] += weight * (float(joint_kl_by_head(native, dropped).mean()) - before_kl)
            upper_drop = joint_distribution(s, mask_k_units(upgraded, [unit], layer), upper_book,
                                            decode_key=ops.decode_exchange_key_fields)
            new_drops[layer][unit] += weight * (float(joint_kl_by_head(native, upper_drop).mean()) - upper_kl)
        if (index + 1) % 28 == 0:
            print(json.dumps({'phase': 'conditional_KL_costs', 'sources_completed': (index + 1) // 28,
                              'expected_sources': 32, 'elapsed_s': time.perf_counter() - started}), flush=True)
    best, candidates = select_profile(base_loss, upgrades, old_drops, new_drops)
    selected_layer, dropped = best['upgraded_layer'], best['dropped_k_units']
    after_loss = 0.
    after_records = []
    for raw in samples:
        s = cal.to_device(raw, device)
        layer = s['layer_idx']
        factors = torch.tensor(s['original_factors']['key_head_factors'][layer], device=device, dtype=torch.float32)
        native = joint_distribution(s, native=True)
        fields, book, _ = ops.precision_exchange_fields(s['r7_fields'], s['visual_k_rotated'], s['book'], high_book, factors,
            layer_idx=layer, upgraded_layer=selected_layer, dropped_k_units=dropped, upgrade=True, compact=False,
            encode_key=callbacks['encode_key'], decode_key=callbacks['decode_key'], encode_sign_depth=callbacks['encode_sign_depth'])
        dist = joint_distribution(s, fields, book, decode_key=ops.decode_exchange_key_fields)
        after_loss += weight * float(joint_kl_by_head(native, dist).mean())
        after_records.append({'source_id': s['source_id'], 'id': s['id'], 'dataset': s['dataset'], 'layer_idx': layer,
            'phase': PHASE, 'step': 0, 'routing': route_metrics(native, dist),
            'projection': projected_arms(s, native, dist, cal.decoded_value(s['r7_fields']), cal._wo(projection_cfg, layer, device))})
    residual = abs(after_loss - best['route_loss_predicted'])
    require(residual <= 1e-8 * (1 + abs(after_loss)), 'four-drop fixed-q separability prediction differs from full replay')
    require(source_before['calibrator'] == sha(__file__) and source_before['exchange_helper'] == sha(ops.__file__) and
            source_before['cal_helper'] == sha(cal.__file__) and source_before['high_book'] == sha(config['high_book_path']) and
            source_before['collection_complete'] == sha(directory / 'complete.json') and
            source_before['collection_manifest'] == sha(directory / 'manifest.json') and
            all(sha(p) == digest for p, digest in callbacks['legacy_source_hashes'].items()), 'source changed during calibration')
    profile = {'schema': 'trajkv.round9.precision_exchange_profile.v1', 'base_profile': 'r7_joint_k1_gt_cond',
        'upgraded_layer': selected_layer, 'dropped_k_units': dropped, 'route_loss_before': base_loss,
        'route_loss_after': after_loss, 'route_loss_predicted_from_conditional_marginals': best['route_loss_predicted'],
        'additive_replay_absolute_residual': residual,
        'chosen_donor_costs': [r for r in best['conditional_donor_costs'] if r['chosen']],
        'all_conditional_donor_costs': best['conditional_donor_costs'],
        'upgrade_keep_all_K_residual_route_delta': best['upgrade_keep_all_K_residual_route_delta'],
        'low_book_sha256': LOW_BOOK_SHA, 'high_book_sha256': HIGH_BOOK_SHA, 'high_book_path': str(config['high_book_path']),
        'base_metadata_factor_sha256': FACTOR_SHA, 'K112_and_V224_factors_unchanged': True,
        'normalization': {'implementation': 'explicit FP64 amax/shift/exp/sum/log',
            'row_sum_atol': ROW_SUM_ATOL, 'parent_v1_sha256': PARENT_V1_SHA,
            'original_KL_chain_and_additivity_thresholds_unchanged': True},
        'original_K_stages': 23, 'retained_K_stages': 19, 'V_stages_unchanged': 20,
        'selection_scope': 'conditional exact minimum over28 single-layer upgrades and four distinct original K stage0 drops, for fixed captured q/text head-separable cal KL only',
        'calibration_phase': PHASE, 'calibration_sources': 32, 'events': len(samples),
        'task_quality_or_global_model_optimum_claimed': False, 'references_read': False,
        'source_provenance': source_before, 'collection_provenance': collection_proof,
        'calibration_ids': sorted({s['id'] for s in samples}), 'source_ids': sorted({s['source_id'] for s in samples}),
        'calibration_source_identities': [dict(source_id=s['source_id'], id=s['id'],
            dataset=s['dataset'], image_sha256=s['image_sha256'], shard_sha256=s['_shard_sha256'])
            for _, s in sorted({s['source_id']: s for s in samples}.items())],
        'private_shard_sha256': sorted({(s['_shard_path'], s['_shard_sha256']) for s in samples})}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'profile.json', profile)
    write(output / 'candidate_profiles.json', candidates)
    for name, rows in [('cal_before', baseline_records), ('cal_selected_after', after_records)]:
        with (output / (name + '.jsonl')).open('x', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + '\n')
    summary = {'schema': SCHEMA, 'status': 'PASS_CAL_ONLY_CONDITIONAL_EXCHANGE_PROFILE',
        'actual_GPU': True, 'GPU_name': torch.cuda.get_device_name(), 'new_model_forwards': 0,
        'calibration_sources': 32, 'source_dataset_counts': {'docvqa': 8, 'textvqa': 16, 'infovqa': 8},
        'phases_used': [PHASE], 'layers': list(range(28)), 'actual_events': len(samples),
        'base_route_loss': base_loss, 'selected_route_loss': after_loss,
        'route_loss_change': after_loss-base_loss, 'cal_route_improved': after_loss < base_loss,
        'selection_objective': 'mean_source mean_layer mean_28_queryheads KL(native visual+same text || candidate visual+same text)',
        'projection_diagnostics_not_selection_objectives': True,
        'projection_arithmetic': 'FP64 dense fixed-q attention, actual sequential-Half reconstructed K/V; output heads Half then frozen Half W_O',
        'normalization': profile['normalization'],
        'cal_before': aggregate(baseline_records), 'cal_selected_after': aggregate(after_records),
        'profile_sha256': sha(output / 'profile.json'), 'candidates_sha256': sha(output / 'candidate_profiles.json'),
        'source_provenance': source_before, 'collection_provenance': collection_proof,
        'physical_deployment_bytes_established': False,
        'calibration_mask_pools_are_not_compacted_or_a_storage_claim': True,
        'task_quality_or_free_generation_claimed': False, 'references_read': False,
        'private_weights_or_PT_must_not_be_published': True, 'wall_s': time.perf_counter() - started}
    write(output / 'summary.json', summary)
    write(output / 'complete.json', {'status': summary['status'], 'actual_GPU': True,
        'calibration_sources': 32, 'actual_events': len(samples), 'new_model_forwards': 0,
        'profile_sha256': summary['profile_sha256'], 'summary_sha256': sha(output / 'summary.json')})
    return summary


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def failure_source_numeric_check(config, output, source_id='10019.jpeg', layer=0):
    """One original failure shard only; no model/WO, profile selection, or task QA.

    The builtin fused CUDA operation is called ONLY here as an observational
    control. It is never used by the production calibration path in version2.
    CPU builtin log_softmax is an independent reference for identical logits.
    """
    require(sha(cal.__file__) == CAL_SHA, 'frozen actual sample loader changed')
    output = Path(output)
    require(not output.exists(), 'immutable numeric-check output already exists')
    directory = Path(config['collection_dir'])
    require(config.get('collection_complete_sha256') == COLLECTION_SHA and
            sha(directory / 'complete.json') == COLLECTION_SHA, 'original real cal32 complete required')
    complete = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
    require(complete['status'] == 'PASS_CALIBRATION_32_REAL_MODEL_SOURCES' and complete['actual_GPU'] is True and
            complete['references_read'] is False, 'actual original collection role')
    sources = [row for row in complete['sources'] if row['source_id'] == source_id or row.get('id') == source_id]
    require(len(sources) == 1 and 0 <= layer < 28, 'one exact failure source/layer required')
    source = sources[0]
    samples = cal.load_sample_shards([source['file']], {source['file']: source['sha256']})
    selected = [s for s in samples if s['layer_idx'] == layer and s['phase'] == PHASE and s['step'] == 0]
    require(len(selected) == 1, 'one actual suffix-last event required')
    s_cpu = selected[0]
    require(s_cpu['q'].shape == (1, 28, 1, 128) and
            s_cpu['student_snapshot'].get('collector_source_sha256') == COLLECTOR_SHA,
            'original compressed student single-query source')
    ops = importlib.import_module('round9_precision_exchange_v1')
    require(config.get('exchange_helper_sha256') == sha(ops.__file__), 'reviewed actual Half decoder required')
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6), 'real SM86 CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = config.get('device', 'cuda:0')
    require(str(device).startswith('cuda'), 'actual CUDA numeric check required')
    s_gpu = cal.to_device(s_cpu, device)
    source_before = {'checker': sha(__file__), 'cal_helper': sha(cal.__file__),
        'exchange_helper': sha(ops.__file__), 'collection_complete': sha(directory / 'complete.json'),
        'one_private_shard': sha(source['file']), 'configuration_canonical_sha256': canonical(config)}
    rows = []
    distributions = {}
    for arm in ('native_K', 'original_low_K'):
        native = arm == 'native_K'
        cpu = joint_distribution(s_cpu, s_cpu['r7_fields'], s_cpu['book'], native=native,
                                 decode_key=ops.decode_exchange_key_fields)
        gpu = joint_distribution(s_gpu, s_gpu['r7_fields'], s_gpu['book'], native=native,
                                 decode_key=ops.decode_exchange_key_fields)
        mask = cpu['visible'][None, None].expand_as(cpu['logp'])
        # Explicitly confined reference/control calls; production uses neither.
        cpu_builtin_reference = torch.nn.functional.log_softmax(cpu['joint_logits'], dim=-1)
        gpu_builtin_observation = torch.nn.functional.log_softmax(gpu['joint_logits'], dim=-1)
        cpu_reference_error = float((cpu['logp'] - cpu_builtin_reference)[mask].abs().max())
        gpu_error = float((gpu['logp'].cpu() - cpu_builtin_reference)[mask].abs().max())
        logits_error = float((gpu['joint_logits'].cpu() - cpu['joint_logits'])[mask].abs().max())
        require(cpu_reference_error <= 1e-12 and gpu_error <= CPU_GPU_LOG_ATOL and logits_error <= CPU_GPU_LOG_ATOL,
                'stable same-source CPU/GPU probability reference mismatch')
        builtin_sum = gpu_builtin_observation.exp().sum(-1)
        rows.append({'arm': arm, 'joint_width': int(cpu['logp'].shape[-1]),
            'same_source_logits_GPU_vs_CPU_maxabs': logits_error,
            'stable_CPU_vs_builtin_CPU_maxabs': cpu_reference_error,
            'stable_GPU_vs_builtin_CPU_maxabs': gpu_error,
            'stable_CPU_row_sum_max_error': check_log_distribution(cpu['logp']),
            'stable_GPU_row_sum_max_error': check_log_distribution(gpu['logp']),
            'builtin_GPU_observation_only_maxabs_vs_CPU': _finite_number(
                (gpu_builtin_observation.cpu() - cpu_builtin_reference)[mask].abs().max()),
            'builtin_GPU_observation_only_exp_row_sum_range':
                [_finite_number(builtin_sum.min()), _finite_number(builtin_sum.max())],
            'builtin_GPU_operation_not_used_for_v2_calibration': True})
        distributions[arm] = (cpu, gpu)
    cpu_route = route_metrics(distributions['native_K'][0], distributions['original_low_K'][0])
    gpu_route = route_metrics(distributions['native_K'][1], distributions['original_low_K'][1])
    require(abs(cpu_route['joint_attention_KL'] - gpu_route['joint_attention_KL']) <= CPU_GPU_LOG_ATOL,
            'same-source joint KL CPU/GPU mismatch')
    require(source_before['checker'] == sha(__file__) and source_before['cal_helper'] == sha(cal.__file__) and
            source_before['exchange_helper'] == sha(ops.__file__) and
            source_before['collection_complete'] == sha(directory / 'complete.json') and
            source_before['one_private_shard'] == sha(source['file']), 'numeric-check source changed')
    result = {'schema': 'trajkv.round9.failure_source_stable_normalization_check.v1',
        'status': 'PASS_ACTUAL_FAILURE_SOURCE_CPU_GPU_STABLE_NORMALIZATION',
        'actual_GPU': True, 'hardware': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'CUDA': torch.version.cuda, 'source_id': s_cpu['source_id'], 'id': s_cpu['id'],
        'requested_source_identifier': source_id, 'layer_idx': layer, 'phase': PHASE,
        'one_original_source_checked': True, 'full_cal32_profile_selected': False,
        'private_shards_loaded': 1, 'model_forwards': 0, 'quality_QA_added': 0,
        'rows': rows, 'CPU_route': cpu_route, 'GPU_route': gpu_route,
        'row_sum_atol': ROW_SUM_ATOL, 'CPU_GPU_log_atol': CPU_GPU_LOG_ATOL,
        'KL_chain_atol_unchanged': 1e-8, 'source_provenance': source_before,
        'private_PT_not_copied_or_published': True, 'references_read': False}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'summary.json', result)
    write(output / 'complete.json', {'status': result['status'], 'actual_GPU': True,
        'one_original_source_checked': True, 'model_forwards': 0, 'quality_QA_added': 0,
        'summary_sha256': sha(output / 'summary.json')})
    return result


def cpu_check(output):
    """Small arithmetic/selection controls, never synthetic complete cal32."""
    torch.manual_seed(9)
    normalization_cases = []
    for width in (29, 796, 797, 798, 799, 800, 1800, 1813):
        logits = torch.randn(1, 28, 1, width, dtype=torch.float64) * 7
        logits[..., -1] = -torch.inf
        manual = stable_log_softmax(logits)
        reference = torch.nn.functional.log_softmax(logits, dim=-1)  # CPU control only.
        finite = torch.isfinite(reference)
        error = float((manual - reference)[finite].abs().max())
        partition_error = float((stable_logsumexp(logits) - torch.logsumexp(logits, -1)).abs().max())
        require(error <= 1e-12 and partition_error <= 1e-12, 'CPU stable normalizer reference mismatch')
        normalization_cases.append({'width': width, 'maxabs_vs_CPU_builtin': error,
            'log_partition_maxabs_vs_CPU_builtin': partition_error,
            'row_sum_max_error': check_log_distribution(manual)})
    rejected_unnormalized = False
    try:
        check_log_distribution(manual + .2)
    except ValueError:
        rejected_unnormalized = True
    require(rejected_unnormalized, 'finite but non-normalized log probabilities accepted')
    invalid_logits_rejected = 0
    for bad in (torch.full((1, 3), -torch.inf, dtype=torch.float64),
                torch.tensor([[0., float('inf')]], dtype=torch.float64),
                torch.tensor([[0., float('nan')]], dtype=torch.float64)):
        try:
            stable_log_softmax(bad)
        except ValueError:
            invalid_logits_rejected += 1
    require(invalid_logits_rejected == 3, 'all-masked/NaN/+Inf logits accepted')
    n, t = 32, 3
    s = {'q': torch.randn(1, 28, 1, 128).half() * .1,
        'q_rotated': torch.randn(1, 28, 1, 128).half() * .1,
        'visual_k': torch.randn(1, 4, n, 128).half() * .1,
        'visual_v': torch.randn(1, 4, n, 128).half() * .1,
        'exact_keys': torch.randn(1, 4, t, 128).half() * .1,
        'exact_values': torch.randn(1, 4, t, 128).half() * .1,
        'visual_positions': torch.arange(n), 'exact_positions': torch.arange(n, n + t),
        'query_positions': torch.tensor([n + t - 1]), 'scaling': 128 ** -.5}
    p = joint_distribution(s, native=True)
    require(float(joint_kl_by_head(p, p).abs().max()) == 0, 'identical full attention KL')
    # A common visual shift leaves conditional visual routing unchanged, but
    # changes its competition with text. The selector must retain this term.
    shifted = dict(p)
    visual = p['visual_logits'] + 1.25
    text = s['q'].double() @ repeat(s['exact_keys']).double().transpose(-1, -2) * s['scaling']
    lp = stable_log_softmax(torch.cat((visual, text), -1), -1)
    shifted.update(visual_logits=visual, logp=lp, p=lp.exp())
    route = route_metrics(p, shifted)
    require(route['joint_attention_KL'] > .001 and abs(route['conditional_visual_KL']) < 1e-12 and
            abs(route['joint_attention_KL'] - route['modality_mass_KL']) < 1e-10,
            'common visual mass error was incorrectly centered away')
    def from_visual(vis):
        logp = stable_log_softmax(torch.cat((vis, text), -1), -1)
        return {**p, 'visual_logits': vis, 'logp': logp, 'p': logp.exp()}
    base_visual = p['visual_logits'] + torch.randn_like(p['visual_logits']) * .15
    base = from_visual(base_visual)
    base_loss = float(joint_kl_by_head(p, base).mean())
    combined = base_visual.clone()
    sum_delta = 0.
    for h in range(4):
        candidate = base_visual.clone()
        delta = torch.randn_like(candidate[:, h*7:(h+1)*7]) * .12
        candidate[:, h*7:(h+1)*7] += delta
        combined[:, h*7:(h+1)*7] += delta
        sum_delta += float(joint_kl_by_head(p, from_visual(candidate)).mean()) - base_loss
    require(abs(float(joint_kl_by_head(p, from_visual(combined)).mean()) - base_loss - sum_delta) < 1e-12,
            'disjoint KV-head donor KL is not additive under fixed q/native target')
    diagnostics = projected_arms(s, p, shifted, (s['visual_v'].float()+.1).half(),
                                 (torch.randn(3, 3584)*.1).half())
    require(abs(diagnostics['Ktrue_target_keeping_V0_MSE'] - diagnostics['MSE_to_trueK_trueV']['CC']) > 1e-8,
            'keeping-V0 target was silently replaced with the true-V target')
    units = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)]
    old = {u: float(i) * .01 for i, u in enumerate(units)}
    upgrades = {l: -.02 * l for l in range(3)}
    new = {l: {u: old[u] - .2 if l == 1 else old[u] + .05 for u in units if u[0] == l} for l in range(3)}
    best, candidates = select_profile(.7, upgrades, old, new)
    brute = []
    for l in upgrades:
        costs = {u: new[l][u] if u[0] == l else old[u] for u in units}
        brute.extend((.7 + upgrades[l] + sum(costs[u] for u in us), l, [list(u) for u in us])
                     for us in itertools.combinations(units, 4))
    require(math.isclose(best['route_loss_predicted'], min(brute)[0], abs_tol=1e-14), 'conditional donor optimum differs from enumeration')
    require(any(x['cost'] < 0 and x['chosen'] for x in best['conditional_donor_costs']), 'negative donor cost must remain legal')
    lk = torch.full((2, 4, 2), -1, dtype=torch.int32)
    lk[0, 1, 0] = 0
    lk[1, 2, 0] = 1
    f = {'depth_residual_lookup': lk, 'depth_residual_code': torch.ones(2, n, 4, dtype=torch.int32)}
    masked = mask_k_units(f, [(2, 1, 0)], 2)
    require(int(masked['depth_residual_lookup'][0, 1, 0]) == -1 and int(lk[0, 1, 0]) == 0 and
            torch.equal(masked['depth_residual_lookup'][1], lk[1]) and
            masked['depth_residual_code'] is f['depth_residual_code'], 'mask modified V/source/pool')
    rejected = False
    try:
        mask_k_units(f, [('v', 2, 1, 0)], 2)
    except ValueError:
        rejected = True
    require(rejected, 'axis-bearing/wrong donor schema must not enter triple profile')
    result = {'status': 'PASS_CPU_EXCHANGE_KL_AND_CONDITIONAL_SELECTION_ONLY',
        'checks': ['full_joint_KL_identity', 'common_visual_shift_kept_as_mass_error',
            'mass_routing_text_KL_chain', 'actual_attention_disjoint_head_KL_additivity',
            'keeping_V0_target_distinct_from_trueV_target_small_projection_fixture',
            'conditional_four_min_donors_equals_bruteforce',
            'negative_drop_marginal_kept', 'K_only_mask_no_V_or_source_mutation', 'bad_donor_schema_rejected',
            'stable_F64_normalization_eight_widths_vs_CPU_builtin',
            'finite_but_unnormalized_probabilities_rejected', 'all_masked_NaN_positiveInf_rejected'],
        'normalization_cases': normalization_cases, 'probability_row_sum_atol': ROW_SUM_ATOL,
        'original_KL_chain_and_replay_thresholds_unchanged': True,
        'source_sha256': sha(__file__), 'actual_GPU_calls': 0, 'real_cal32_profile_selected': False,
        'task_quality_or_storage_claim': False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'CPU_RECEIPT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--cpu-check', action='store_true')
    action.add_argument('--numeric-check', action='store_true')
    parser.add_argument('--failure-source-id', '--source-id', default='10019.jpeg')
    parser.add_argument('--failure-layer', '--layer', type=int, default=0)
    args = parser.parse_args()
    if args.cpu_check:
        result = cpu_check(args.output)
    else:
        require(args.config is not None, 'explicit calibration config required')
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        result = failure_source_numeric_check(config, args.output, args.failure_source_id, args.failure_layer) \
            if args.numeric_check else calibrate(config, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'profile_selected', 'actual_GPU', 'actual_events',
                       'base_route_loss', 'selected_route_loss', 'profile_sha256', 'source_id',
                       'layer_idx', 'one_original_source_checked') if k in result}), flush=True)


if __name__ == '__main__':
    main()
