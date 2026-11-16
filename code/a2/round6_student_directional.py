"""Round6 native Full/student x CE/fixed-margin directional bridge.

No gold answers are accessed. Each finite intervention creates a clean prefix
cache and replays the entire question suffix and identical teacher-token history.
The codec/native-coordinate bridge is calibration evidence, not packed accuracy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
from bank import install_legacy
from round6_student_codec import (sha, selected_rows, settings_from_config, select_units,
    encode_layer, make_entries, native_replay, teacher_case)


def objective(logits, case, name, position=None):
    if name == 'ce':
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), case['targets'].reshape(-1))
    if name == 'margin':
        p = int(position)
        return logits[0, p, case['targets'][0, p]].float() - logits[0, p, case['competitors'][0, p]].float()
    raise ValueError('Objectives must be ce or margin')


def action_list(config, selected, nl, nh):
    """Active units are explicit one-stage removals; inactive units add a stage."""
    actions = []
    for axis in ('k', 'v'):
        for layer in range(nl):
            for head in range(nh):
                unit = (axis, layer, head)
                before = int(unit in selected)
                actions.append({'axis': axis, 'layer': layer, 'head': head,
                    'before_stage': before, 'after_stage': 1 - before,
                    'action': 'remove_sign_stage' if before else 'add_sign_stage'})
    requested = config.get('finite_interventions')
    if requested is None:
        finite = []
        for axis in ('k', 'v'):
            for before in (0, 1):
                group = [a for a in actions if a['axis'] == axis and a['before_stage'] == before]
                if group:
                    finite.extend(group[:2] if before == 0 else group[:1])
    else:
        finite = []
        for item in requested:
            axis, layer, head = ((item['axis'], int(item['layer']), int(item['head']))
                if isinstance(item, dict) else (item[0], int(item[1]), int(item[2])))
            matches = [a for a in actions if (a['axis'], a['layer'], a['head']) == (axis, layer, head)]
            if len(matches) != 1:
                raise ValueError('Invalid finite intervention')
            finite.append(matches[0])
    return actions, finite


def changed_selection(selected, action):
    result = set(selected)
    unit = (action['axis'], action['layer'], action['head'])
    result.add(unit) if action['after_stage'] else result.discard(unit)
    return result


@torch.no_grad()
def concentration_diagnostics(model, case, entries):
    """At answer queries only: full-softmax visual a, total o including text.

    Only O(answer_queries * total_cached_tokens) weights are transiently formed;
    no visual-token N x N statistic is constructed or saved.
    """
    from calibrate_gradient_rank_round5 import make_prefix_cache
    from p184_working_set import attention
    cache, _, _ = make_prefix_cache(entries, model.config.get_text_config(), False)
    modules = [layer.self_attn for layer in model.model.language_model.layers]
    rows, target_count = [], case['targets'].shape[1]
    def read(q, index):
        k, v = cache.layers[index].keys, cache.layers[index].values
        group = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        positions = torch.arange(k.shape[-2], device=q.device)
        mask = positions[None, None, None, :] <= case['qpos'][:, None, :, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        aq = q[:, :, -target_count:].float()
        scores = aq @ k.float().transpose(-1, -2) / q.shape[-1] ** .5
        weights = scores.masked_fill(~mask[:, :, -target_count:], -torch.inf).softmax(-1)
        av = weights.index_select(-1, case['visual'])
        total = weights @ v.float()
        vv = v.index_select(2, case['visual']).float()
        av2 = av.square()
        # Algebraic expansion avoids an answer_queries x visual_tokens x D owner.
        ct = ((av2 @ vv.square().sum(-1, keepdim=True)).squeeze(-1)
              - 2 * ((av2 @ vv) * total).sum(-1)
              + av2.sum(-1) * total.square().sum(-1)).clamp_min(0)
        rows.append({'layer': index, 'C_sum_visual_a2': av.square().sum(-1).cpu().tolist(),
            'Ctilde_sum_visual_a2_value_minus_total_o_norm2': ct.cpu().tolist(),
            'visual_mass': av.sum(-1).cpu().tolist()})
        return out
    for index, module in enumerate(modules):
        module._p185_read = lambda q, index=index: read(q, index)
    try:
        with attention(model):
            model(input_ids=case['suffix'], position_ids=case['qpos'], past_key_values=cache,
                  use_cache=True, return_dict=True, logits_to_keep=target_count)
    finally:
        for module in modules:
            del module._p185_read
    return rows


def finite_metrics(rows):
    groups = {}
    for row in rows:
        groups.setdefault((str(row['id']), row['objective'], row.get('position'), row['delta_bytes']), []).append(row)
    metrics = []
    for (case_id, obj, position, delta_bytes), group in groups.items():
        predicted = torch.tensor([r['predicted_improvement'] for r in group], dtype=torch.float64)
        actual = torch.tensor([r['actual_improvement'] for r in group], dtype=torch.float64)
        pick = int(predicted.argmax())
        pairs = [(i, j) for i in range(len(group)) for j in range(i) if actual[i] != actual[j]]
        metrics.append({'id': case_id, 'objective': obj, 'position': position, 'delta_bytes': delta_bytes,
            'actions': len(group), 'same_marginal_byte_cost': True,
            'sign_hit_rate': float(((predicted > 0) == (actual > 0)).double().mean()),
            'mean_absolute_directional_error': float((predicted - actual).abs().mean()),
            'within_question_pairwise_order_accuracy': (sum(bool((predicted[i]-predicted[j])*(actual[i]-actual[j]) > 0) for i,j in pairs) / len(pairs)) if pairs else None,
            'selected_action_regret': float(actual.max() - actual[pick]),
            'selected_action': group[pick]['action'],
            'R2_diagnostic': float(1 - (actual-predicted).square().sum()/(actual-actual.mean()).square().sum()) if float((actual-actual.mean()).square().sum()) else None})
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    a = c['arguments']
    install_legacy(a['legacy_scripts'])
    from p180_multiturn_core import attention_bridge
    import run_p175_modern_replication as oldbase
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = settings_from_config(c)
    base = json.loads(Path(settings['metadata_factor_path']).read_text())
    if base['encoder_book_sha256'] != sha(settings['direction_book_path']) or base['value_group_size'] != settings['v_group_size']:
        raise ValueError('Frozen base/book/window mismatch')
    rows = selected_rows(json.loads(Path(a['samples']).read_text(encoding='utf-8')),
        int(c.get('calibration_image_count', 16)), int(c.get('max_cases', 16)), bool(c.get('first_question_per_image', True)))
    if not rows:
        raise ValueError('No calibration cases')
    manifest = {'schema': 'trajkv_round6_student_directional_v1', 'config': c,
        'config_sha256': sha(args.config), 'samples_sha256': sha(a['samples']),
        'encoder_book_sha256': sha(settings['direction_book_path']), 'metadata_sha256': sha(settings['metadata_factor_path']),
        'selected_case_ids': [r['id'] for r in rows], 'gold_answers_loaded': False,
        'teacher_targets': 'Full self-generated; fixed teacher target vs fixed Full best non-target competitor',
        'gradient_scope': 'native SDPA; detached full or decoded mixed prefix KV leaves; actual signed codec delta',
        'compression_boundary': 'after final visual token; entire first-question suffix and identical teacher history replayed',
        'finite_scope': 'fresh student prefix DynamicCache for every single-axis/head stage addition/removal',
        'native_vs_packed_limit': 'materialized K-coordinate bridge omits packed reader WHT(q) half-rounding; native finite replay is not serving accuracy',
        'sources': {x.name: sha(x) for x in (Path(__file__), Path(__file__).with_name('round6_student_codec.py'))}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    processor = AutoProcessor.from_pretrained(a['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_config = model.config.get_text_config()
    nl, nh = text_config.num_hidden_layers, text_config.num_key_value_heads
    book = torch.tensor(json.loads(Path(settings['direction_book_path']).read_text())['centers'], device=model.device, dtype=torch.float16)
    scale = float(c.get('gradient_loss_scale', 128.))
    if not scale > 0 or not 1 <= int(c.get('target_tokens', 4)) <= 8:
        raise ValueError('Positive gradient scale and 1..8 teacher tokens required')
    score_rows, finite_rows, control_rows = [], [], []
    with (args.output/'cases.jsonl').open('w') as stream:
        for number, row in enumerate(rows):
            started = time.perf_counter()
            case = teacher_case(model, processor, row, a, int(c.get('target_tokens', 4)), attention_bridge, oldbase.stop_ids(model, processor))
            selected, budget = select_units(settings, base, nl, nh, case['visual'].numel())
            selected = set(selected)
            vg = case['visual'].numel() if settings['v_group_size'] == 'all' else int(settings['v_group_size'])
            encoded = [encode_layer(k.index_select(2, case['visual']), v.index_select(2, case['visual']), book, base, i, vg)
                       for i, (k, v) in enumerate(case['prefix'])]
            student = make_entries(case['prefix'], encoded, case['visual'], selected)
            actions, finite = action_list(c, selected, nl, nh)
            deltas = {}
            with torch.no_grad():
                for action in actions:
                    layer, head, axis = action['layer'], action['head'], action['axis']
                    from round6_student_codec import decoded_layer
                    changed = decoded_layer(encoded[layer], layer, changed_selection(selected, action))[0 if axis == 'k' else 1]
                    current = student[layer][0 if axis == 'k' else 1].index_select(2, case['visual'])
                    deltas[(axis, layer, head)] = changed[:, head].float() - current[:, head].float()
                baseline, _, _ = native_replay(model, case, student, attention_bridge)
                baseline = baseline.detach().float()
            positions = [int(x) for x in c.get('target_positions', [0]) if int(x) < case['targets'].shape[1]]
            specs = [('ce', None)] if 'ce' in c.get('objectives', ['ce', 'margin']) else []
            if 'margin' in c.get('objectives', ['ce', 'margin']):
                specs += [('margin', position) for position in positions]
            predictions = {}
            for state in c.get('states', ['full', 'student']):
                if state not in ('full', 'student'):
                    raise ValueError('States must be full or student')
                for name, position in specs:
                    logits, leaf_k, leaf_v = native_replay(model, case, case['prefix'] if state == 'full' else student, attention_bridge, True)
                    loss = objective(logits, case, name, position)
                    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                        raise RuntimeError('Detached/nonfinite native objective')
                    (loss * scale).backward()
                    for action in actions:
                        axis, layer, head = action['axis'], action['layer'], action['head']
                        grad = (leaf_k if axis == 'k' else leaf_v)[layer].grad
                        if grad is None:
                            raise RuntimeError('Missing native-cache gradient')
                        grad = grad.index_select(2, case['visual'])[:, head].float() / scale
                        if not bool(torch.isfinite(grad).all()):
                            raise RuntimeError('Nonfinite native-cache gradient')
                        directional = float((grad * deltas[(axis, layer, head)]).sum())
                        record = {'id': row['id'], 'image_id': row['image_id'], 'state': state,
                            'objective': name, 'position': position, **action,
                            'signed_g_dot_actual_codec_delta': directional,
                            'predicted_improvement': directional if name == 'margin' else -directional,
                            'delta_l2': float(deltas[(axis, layer, head)].norm()), 'objective_value': float(loss.detach())}
                        score_rows.append(record)
                        predictions[(state, name, position, axis, layer, head)] = record
                    if state == 'student':
                        diff = logits.detach().float() - baseline
                        control_rows.append({'id': row['id'], 'objective': name, 'position': position,
                            'gradient_vs_no_grad_max_abs': float(diff.abs().max()),
                            'gradient_vs_no_grad_relative_l2': float(diff.norm()/baseline.norm().clamp_min(1e-20))})
                    elif state == 'full':
                        diff = logits.detach().float().cpu() - case['teacher_logits']
                        control_rows.append({'id': row['id'], 'objective': name, 'position': position,
                            'full_same_input_replay_vs_teacher_generation_max_abs': float(diff.abs().max()),
                            'full_same_input_argmax_same': bool(torch.equal(logits.detach().argmax(-1).cpu(), case['teacher_logits'].argmax(-1)))})
                    del logits, leaf_k, leaf_v, loss
            with torch.no_grad():
                for action in finite:
                    repaired = make_entries(case['prefix'], encoded, case['visual'], changed_selection(selected, action))
                    logits, _, _ = native_replay(model, case, repaired, attention_bridge)
                    for name, position in specs:
                        before, after = objective(baseline, case, name, position), objective(logits, case, name, position)
                        pred = predictions.get(('student', name, position, action['axis'], action['layer'], action['head']))
                        if pred is None:
                            continue
                        signed = float(after - before)
                        finite_rows.append({'id': row['id'], 'image_id': row['image_id'], 'objective': name,
                            'position': position, 'action': action, 'before': float(before), 'after': float(after),
                            'delta_bytes': (action['after_stage']-action['before_stage'])*budget['unit_bytes'],
                            'actual_signed_change': signed, 'actual_improvement': signed if name == 'margin' else -signed,
                            'predicted_improvement': pred['predicted_improvement'],
                            'signed_g_dot_actual_codec_delta': pred['signed_g_dot_actual_codec_delta'], 'fresh_cache': True})
                    del repaired, logits
            diagnostics = concentration_diagnostics(model, case, student) if number < int(c.get('concentration_cases', 1)) else []
            torch.cuda.synchronize()
            log = {'id': row['id'], 'image_id': row['image_id'], 'case': number + 1,
                'teacher_token_ids': case['targets'].tolist(), 'fixed_competitor_ids': case['competitors'].tolist(),
                'visual_tokens': case['visual'].numel(), 'question_suffix_tokens_replayed': case['prompt_suffix_tokens'],
                'budget': budget, 'C_Ctilde_diagnostics': diagnostics, 'seconds': time.perf_counter() - started}
            stream.write(json.dumps(log)+'\n'); stream.flush()
            print(json.dumps({k:v for k,v in log.items() if k not in ('C_Ctilde_diagnostics','budget')}), flush=True)
            del case, encoded, student, deltas, baseline
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Base model was modified')
    result = {'schema': manifest['schema'], 'status': 'COMPLETE', 'cases': len(rows),
        'labels_used': False, 'model_weights_frozen': True, 'directional_scores': score_rows,
        'finite_interventions': finite_rows, 'finite_metrics': finite_metrics(finite_rows),
        'native_cache_controls': control_rows, 'task_accuracy_eligible': False,
        'R2_used_as_gate': False, 'independent_same_text_replays': True}
    (args.output/'directional_scores.json').write_text(json.dumps(result, indent=2))
    (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','cases':len(rows)}))


if __name__ == '__main__':
    main()
