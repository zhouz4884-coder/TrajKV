"""RateQuant-style strategy port: actual native-coordinate D(depth) x CE w.

This is not the official RateQuant implementation. It reuses the frozen
Round5 pooled-answer CE sensitivity, retains its common K/V scale, measures
the current TrajKV packed codec's actual 0/1/2-stage distortion on cal16,
and greedily emits a complete legal stage-prefix order. No task score is fit.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def greedy_stage_order(curves, sensitivities):
    """Equal-byte stages: choose highest available actual DeltaD*w.

    Stage zero becomes eligible initially, stage one only after its own zero.
    Both positive and negative measured gains remain in the complete order.
    Ties use axis/layer/head/stage, independent of process/hash iteration order.
    """
    if set(curves) != {'k', 'v'} or set(sensitivities) != {'k', 'v'}:
        raise ValueError('Both K and V curves/sensitivities are required')
    layers, heads = len(curves['k']), len(curves['k'][0])
    heap, gains, trace, selected = [], {}, [], []
    for axis in ('k', 'v'):
        if len(curves[axis]) != layers or len(sensitivities[axis]) != layers:
            raise ValueError('Layer count mismatch')
        for layer in range(layers):
            if len(curves[axis][layer]) != heads or len(sensitivities[axis][layer]) != heads:
                raise ValueError('KV-head count mismatch')
            for head in range(heads):
                ds, weight = curves[axis][layer][head], float(sensitivities[axis][layer][head])
                if len(ds) != 3 or any(not math.isfinite(float(d)) or d < 0 for d in ds):
                    raise ValueError('Each unit needs finite nonnegative native MSE D(0/1/2)')
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError('CE sensitivity must be finite and nonnegative')
                for stage in (0, 1):
                    gains[(axis, layer, head, stage)] = weight * (float(ds[stage]) - float(ds[stage + 1]))
                heapq.heappush(heap, (-gains[(axis, layer, head, 0)], axis, layer, head, 0))
    cumulative = 0.
    while heap:
        minus_score, axis, layer, head, stage = heapq.heappop(heap)
        action = (axis, layer, head, stage)
        selected.append(action)
        score = -minus_score
        cumulative += score
        trace.append(dict(rank=len(selected), axis=axis, layer=layer, head=head, stage=stage,
            sensitivity_w=float(sensitivities[axis][layer][head]),
            native_mse_before=float(curves[axis][layer][head][stage]),
            native_mse_after=float(curves[axis][layer][head][stage + 1]),
            weighted_mse_gain=score, cumulative_weighted_mse_gain=cumulative))
        if stage == 0:
            following = (axis, layer, head, 1)
            heapq.heappush(heap, (-gains[following], *following))
    if len(selected) != 4 * layers * heads or len(set(selected)) != len(selected):
        raise RuntimeError('Complete 0/1 stage action space was not emitted exactly once')
    seen = set()
    for axis, layer, head, stage in selected:
        if stage and (axis, layer, head, 0) not in seen:
            raise RuntimeError('Greedy stage-prefix violation')
        seen.add((axis, layer, head, stage))
    return selected, trace


def cpu_control():
    # The initially largest second gain must wait for its own first stage.
    curves = {'k': [[[10., 9., 0.], [20., 10., 9.]]],
              'v': [[[4., 2., 1.], [3., 4., 0.]]]}
    weights = {'k': [[1., .5]], 'v': [[1., 1.]]}
    order, trace = greedy_stage_order(curves, weights)
    expected = [('k', 0, 1, 0), ('v', 0, 0, 0), ('k', 0, 0, 0), ('k', 0, 0, 1),
                ('v', 0, 0, 1), ('k', 0, 1, 1), ('v', 0, 1, 0), ('v', 0, 1, 1)]
    if order != expected or sum(row['weighted_mse_gain'] < 0 for row in trace) != 1:
        raise RuntimeError('Prefix/decreasing/nonmonotonic gain CPU control failed')
    # Independent eligible-set oracle, including delayed unlock and ties.
    seen = set()
    for action in order:
        candidates = []
        for axis in ('k', 'v'):
            for head in range(2):
                for stage in (0, 1):
                    item = (axis, 0, head, stage)
                    if item in seen or stage and (axis, 0, head, 0) not in seen:
                        continue
                    ds = curves[axis][0][head]
                    score = weights[axis][0][head] * (ds[stage] - ds[stage + 1])
                    candidates.append((-score, *item))
        if action != tuple(min(candidates)[1:]):
            raise RuntimeError('Heap ranking disagrees with independent legal-action oracle')
        seen.add(action)
    return dict(status='PASS', actions=8, prefix_violations=0,
        nonmonotonic_measured_gain_preserved=True,
        scope='CPU stage allocation only; actual codec/calibration requires GPU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    control = cpu_control()
    write_json(args.output / 'cpu_control.json', control)
    if args.self_test:
        print(json.dumps(control), flush=True)
        return
    if args.config is None:
        parser.error('--config is required unless --self-test is used')
    import torch
    from bank import install_legacy
    from round6_student_codec import settings_from_config, selected_rows, teacher_case, encode_layer
    from round6_depth_codec import encode_sign_depth, decode_sign_stage, validate_units, validate_stages
    from fast_wht import wht, torch_wht
    if not torch.cuda.is_available():
        raise RuntimeError('Actual Full prefill/FP16 depth calibration requires CUDA')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    a, settings = config['arguments'], settings_from_config(config)
    install_legacy(a['legacy_scripts'])
    from p180_multiturn_core import attention_bridge
    import run_p175_modern_replication as oldbase
    from transformers import AutoProcessor, AutoModelForImageTextToText
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    gradient_path = Path(config['gradient_ranking_path'])
    gradient = json.loads(gradient_path.read_text(encoding='utf-8-sig'))
    book_path, factor_path = Path(settings['direction_book_path']), Path(settings['metadata_factor_path'])
    base = json.loads(factor_path.read_text())
    if base['encoder_book_sha256'] != sha(book_path) or base['value_group_size'] != settings['v_group_size']:
        raise ValueError('Frozen encoder/base/window mismatch')
    for key, digest in (('encoder_book_sha256', sha(book_path)), ('fixed_metadata_sha256', sha(factor_path))):
        if gradient.get(key) != digest:
            raise ValueError('Reused CE sensitivities have a different frozen source: ' + key)
    if gradient.get('K_V_gradient_sensitivities_separately_normalized') is not False:
        raise ValueError('Reused sensitivities must explicitly preserve the common K/V CE scale')
    count = int(config.get('calibration_image_count', 16))
    if count != 16:
        raise ValueError('This frozen calibration is exactly cal16 unique images')
    rows = selected_rows(json.loads(Path(a['samples']).read_text(encoding='utf-8-sig')), count, count, True)
    if len(rows) != count or len({r['image_id'] for r in rows}) != count:
        raise ValueError('Need exactly cal16 unique image first questions')
    manifest = dict(schema='trajkv_round6_depth_rd_manifest_v1', config=config,
        config_sha256=sha(args.config), samples_sha256=sha(a['samples']),
        gradient_ranking_sha256=sha(gradient_path), encoder_book_sha256=sha(book_path),
        fixed_metadata_sha256=sha(factor_path),
        calibration_case_ids=[row['id'] for row in rows], calibration_image_ids=[row['image_id'] for row in rows],
        first_question_per_image=True, target_tokens_for_teacher_case=1,
        forward_scope='one Full native prefill per unique image first question; no answer-token replay/backward',
        labels_used=False, task_scores_fit=False, model_weights_frozen=True,
        sensitivity_source='frozen round5 per_axis.{k,v}.sensitivity_w pooled answer CE gradient energy',
        K_V_gradient_sensitivities_separately_normalized=False,
        distortion_scope='mean over cal16 per-image per-scalar MSE in native K/V coordinates after actual FP16 stage reconstruction',
        K_native_bridge_limit='native inverse-WHT-plus-mean FP16 reconstruction omits deployed query-WHT FP16 rounding',
        reference_method='RateQuant-style sensitivity x actual distortion stage allocation strategy-port',
        official_reproduction=False,
        differences_from_official='existing pooled DocVQA self-target CE sensitivities; actual TrajKV fixed sign stages; legal-prefix greedy equal-byte order',
        sources={name: sha(Path(__file__).with_name(name)) for name in
            ('round6_fit_depth_rd.py', 'round6_student_codec.py', 'round6_depth_codec.py', 'quantizers.py', 'fast_wht.py')})
    write_json(args.output / 'manifest.json', manifest)
    processor = AutoProcessor.from_pretrained(a['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_config = model.config.get_text_config()
    nl, nh = text_config.num_hidden_layers, text_config.num_key_value_heads
    if text_config.hidden_size // text_config.num_attention_heads != 128:
        raise ValueError('D128 model required')
    weights = {axis: gradient['per_axis'][axis]['sensitivity_w'] for axis in ('k', 'v')}
    for axis in ('k', 'v'):
        tensor = torch.tensor(weights[axis], dtype=torch.float64)
        if tuple(tensor.shape) != (nl, nh) or not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError('Invalid common-scale CE sensitivity shape/value')
    incumbent_units = validate_units(settings['selected_units'], nl, nh)
    if len(incumbent_units) != 2 * nl * nh:
        raise ValueError('Incumbent envelope requires its complete 2*L*H unit ranking')
    envelope = [(*unit, stage) for stage in (0, 1) for unit in incumbent_units]
    validate_stages(envelope, nl, nh)
    book = torch.tensor(json.loads(book_path.read_text())['centers'], device=model.device, dtype=torch.float16)
    if tuple(book.shape) != (256, 4):
        raise ValueError('Frozen global encoder book must be [256,4]')
    totals = {axis: torch.zeros((nl, nh, 3), dtype=torch.float64) for axis in ('k', 'v')}
    stats = []
    stops = oldbase.stop_ids(model, processor)
    with (args.output / 'cases.jsonl').open('w', encoding='utf-8') as stream, torch.inference_mode():
        for index, row in enumerate(rows):
            started = time.perf_counter()
            case = teacher_case(model, processor, row, a, 1, attention_bridge, stops)
            visual = case['visual']
            nv = int(visual.numel())
            vg = nv if settings['v_group_size'] == 'all' else int(settings['v_group_size'])
            case_curves = {axis: torch.empty((nl, nh, 3), dtype=torch.float64) for axis in ('k', 'v')}
            for layer, (prefix_k, prefix_v) in enumerate(case['prefix']):
                sk, sv = prefix_k.index_select(2, visual), prefix_v.index_select(2, visual)
                entry = encode_layer(sk, sv, book, base, layer, vg)
                original_mean = sk.float().mean(-2, keepdim=True).half()
                source_rotated = wht((sk.float() - original_mean.float()).contiguous())
                for axis, source, initial, stage_target in (
                        ('k', sk, entry['krot'], source_rotated),
                        ('v', sv, entry['v_base'], sv)):
                    codes, alphas, _, _ = encode_sign_depth(stage_target, initial, 2)
                    actual = initial
                    for depth in (0, 1, 2):
                        if depth:
                            actual = decode_sign_stage(actual, codes[depth - 1], alphas[depth - 1])
                        native = (torch_wht(actual.float()) + entry['mean'].float()).half() if axis == 'k' else actual
                        mse = (native.float() - source.float()).square().mean(dim=(0, 2, 3)).double().cpu()
                        case_curves[axis][layer, :, depth] = mse
            for axis in ('k', 'v'):
                totals[axis] += case_curves[axis]
            torch.cuda.synchronize()
            log = dict(case=index + 1, id=row['id'], image_id=row['image_id'], visual_tokens=nv,
                prefix_boundary=case['boundary'], first_question_only=True,
                source_mse_by_axis_layer_head_depth={axis: value.tolist() for axis, value in case_curves.items()},
                seconds=time.perf_counter() - started)
            stream.write(json.dumps(log) + '\n')
            stream.flush()
            stats.append({key: value for key, value in log.items() if key != 'source_mse_by_axis_layer_head_depth'})
            print(json.dumps(dict(phase='depth_rd_actual_curves', case=index + 1, cases=len(rows),
                id=row['id'], visual_tokens=nv, seconds=round(log['seconds'], 2))), flush=True)
            del case, case_curves, visual, entry, sk, sv, source_rotated, codes, alphas, native, actual
    curves = {axis: (totals[axis] / len(rows)).tolist() for axis in ('k', 'v')}
    order, trace = greedy_stage_order(curves, weights)
    validate_stages(order, nl, nh)
    common = dict(schema='trajkv_round6_depth_priority_v1', status='COMPLETE',
        stage_index='zero-based 0/1; each stage1 follows its own stage0', layers=nl, kv_heads=nh,
        stages=len(order), ranking_bytes=2 * len(order), descriptor_bytes=5,
        native_distortion_coordinate_scope=manifest['distortion_scope'],
        gradient_ranking_sha256=sha(gradient_path), samples_sha256=sha(a['samples']),
        calibration_case_ids=manifest['calibration_case_ids'], calibration_image_ids=manifest['calibration_image_ids'],
        encoder_book_sha256=sha(book_path), fixed_metadata_sha256=sha(factor_path),
        manifest_sha256=sha(args.output / 'manifest.json'), K_V_gradient_sensitivities_separately_normalized=False,
        official_reproduction=False, task_accuracy_claim=False)
    write_json(args.output / 'policy.json', dict(common, strategy='ratequant_style_depth_rd_strategy_port',
        selected_stages=order, priority_formula='sensitivity_w[axis,L,H] * mean_cal16(D_native(stage)-D_native(stage+1)); legal-prefix greedy',
        trace=trace))
    write_json(args.output / 'incumbent_envelope.json', dict(common,
        strategy='incumbent_complete_width_then_depth_envelope', selected_stages=envelope))
    write_json(args.output / 'curves.json', dict(schema='trajkv_round6_actual_native_depth_curves_v1',
        per_axis_native_mse=curves, per_axis_sensitivity_w=weights, cases=stats,
        weighting='unweighted mean over unique calibration images of per-image per-scalar native MSE',
        scope='distortion diagnostic/allocation proxy; no ANLS, accuracy or free-answer quality claim'))
    flat = []
    for axis in ('k', 'v'):
        for layer in range(nl):
            for head in range(nh):
                ds, weight = curves[axis][layer][head], float(weights[axis][layer][head])
                flat.append(dict(axis=axis, layer=layer, head=head, sensitivity_w=weight,
                    D0=ds[0], D1=ds[1], D2=ds[2], D1_over_D0=ds[1] / max(ds[0], 1e-30),
                    D2_over_D1=ds[2] / max(ds[1], 1e-30),
                    weighted_stage0_gain=weight * (ds[0] - ds[1]), weighted_stage1_gain=weight * (ds[1] - ds[2])))
    with (args.output / 'curves.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    write_json(args.output / 'complete.json', dict(status='PASS', calibration_images=len(rows),
        layers=nl, kv_heads=nh, stages=len(order), legal_prefixes=True,
        ranking_bytes=2 * len(order), descriptor_bytes=5, model_weights_frozen=True,
        policy_sha256=sha(args.output / 'policy.json'), incumbent_envelope_sha256=sha(args.output / 'incumbent_envelope.json'),
        curves_sha256=sha(args.output / 'curves.json'), cpu_control=control))
    print(json.dumps(dict(status='PASS', calibration_images=len(rows), stages=len(order),
        ranking_bytes=2 * len(order), descriptor_bytes=5)), flush=True)


if __name__ == '__main__':
    main()
