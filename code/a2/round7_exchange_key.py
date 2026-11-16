"""Round7 v4: finite exchanges with real packed Depth or G4_16/G4_256 KeyBank.

This is cached-prefix calibration, not end-to-end quality/timing evidence. Each
intervention and each target sequence gets a fresh native prefix DynamicCache,
a freshly encoded packed bank, and a clean question/teacher-history replay.
No materialized codec perturbation, gradient, or serving reference owner is used.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from round7_exchange_policy import (answer, apply_swaps, candidates, choose_rows,
    load_references, macro_scores, pick_disjoint, ranking_units, sha, source_id,
    stages, unit)

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def depth_plan(settings, base, layers, heads, visual_tokens, explicit_ranking=None,
               stage_limit=None, budget_bits=None):
    from round6_depth_codec import plan_depth_stages
    key_method = settings.get('method', 'round6_depth') == 'round7_key'
    book_bytes, index_bytes, metric_bytes = 2048, 32, 0
    if key_method:
        centers = json.loads(Path(settings['direction_book_path']).read_text())['centers']
        if len(centers) not in (16, 256) or any(len(value) != 4 for value in centers):
            raise ValueError('KeyBank requires an actual 16x4 or 256x4 book')
        book_bytes, index_bytes = len(centers) * 4 * 2, (16 if len(centers) == 16 else 32)
        if settings.get('key_metric_path'):
            from round7_key_artifacts import load_rank1_metric
            metric_u, metric_gamma, _, _ = load_rank1_metric(settings['key_metric_path'], layers, heads, 'cpu')
            metric_bytes = metric_u.numel() * metric_u.element_size() + metric_gamma.numel() * metric_gamma.element_size()
    group = visual_tokens if settings.get('v_group_size', 'all') == 'all' else int(settings['v_group_size'])
    full = layers * heads * visual_tokens * 128 * 4
    base_bytes = (layers * heads * visual_tokens * (index_bytes + 2 + 16) + layers * heads * 128 * 2
                  + 4 * layers * heads * 128 * (visual_tokens // group) + book_bytes + 8 * visual_tokens)
    if stage_limit is None:
        stage_limit = settings.get('res_stages_limit', settings.get('res_units_limit', settings.get('residual_units_limit')))
    if explicit_ranking is None and settings.get('selected_stages') is not None:
        explicit_ranking = [tuple(value[:3]) for value in settings['selected_stages']]
    if key_method and explicit_ranking is None:
        # K1 has room for many more stages. Never accidentally enter the old
        # named-K preservation policy when a fixed explicit stage0 experiment
        # was requested.
        explicit_ranking = [unit(value) for value in settings['selected_units']]
    result = plan_depth_stages(layers=layers, heads=heads, visual_tokens=visual_tokens,
        visual_full_bytes=full, base_bytes=base_bytes, factor_bytes=4 * int(base['parameter_count']) + metric_bytes,
        budget_bits=float(settings.get('budget_bits', 1.8) if budget_bits is None else budget_bits),
        selected_units=settings.get('selected_units', []),
        selected_stages=stages(explicit_ranking) if explicit_ranking is not None else None,
        depth_policy='incumbent', stage_limit=stage_limit)
    result.update(key_index_bytes_per_token=index_bytes, book_bytes=book_bytes,
        factor_parameter_bytes=4 * int(base['parameter_count']), metric_parameter_bytes=metric_bytes,
        base_payload_and_metadata_bytes=base_bytes)
    if settings.get('res_stages_limit') is not None and stage_limit is not None and len(result['selected_stages']) != int(stage_limit):
        raise RuntimeError('Requested fixed stage count does not fit this actual packed complete-owner budget')
    return result


def intervention_settings(settings, base, layers, heads, visual_tokens, ranking, operation, pair):
    current = depth_plan(settings, base, layers, heads, visual_tokens, ranking)
    count = len(current['selected_stages'])
    active = {tuple(value[:3]) for value in current['selected_stages']}
    ordered, cap, budget = list(ranking), count, float(settings.get('budget_bits', 1.8))
    if operation != 'baseline':
        remove, add = unit(pair['remove']), unit(pair['add'])
        if remove not in active or add in active:
            raise ValueError('A frozen candidate is not legal in this current student state')
        if operation == 'joint':
            ordered = apply_swaps(ranking, [pair])
        elif operation == 'remove':
            ordered = [value for value in ranking if value != remove] + [remove]
            cap = count - 1  # Prevent the next cold unit from replacing the removal.
        elif operation == 'add':
            ordered = [value for value in ranking if value != add]
            ordered.insert(count, add)
            cap = count + 1
            full = layers * heads * visual_tokens * 128 * 4
            if current['fixed_bytes'] + cap * current['stage_bytes'] > current['allowed_bytes']:
                budget = math.nextafter(16 * (current['allowed_bytes'] + current['stage_bytes']) / full, math.inf)
        else:
            raise ValueError('Unknown finite intervention')
    result = dict(settings)
    result.pop('selected_units', None)
    result.pop('depth_policy', None)
    result.pop('selected_stages', None)
    for key in ('res_units_limit', 'residual_units_limit', 'res_stages_limit'):
        result.pop(key, None)
    result.update(selected_stages=stages(ordered), res_stages_limit=cap, budget_bits=budget)
    planned = depth_plan(settings, base, layers, heads, visual_tokens, ordered, cap, budget)
    chosen = {tuple(value[:3]) for value in planned['selected_stages']}
    expected = active.copy()
    if operation in ('remove', 'joint'):
        expected.remove(unit(pair['remove']))
    if operation in ('add', 'joint'):
        expected.add(unit(pair['add']))
    if chosen != expected or len(result['selected_stages']) != 2 * layers * heads:
        raise RuntimeError('Singleton or joint allocation changed an unintended unit/ranking charge')
    return result, planned


def load_settings(config):
    if config.get('incumbent_config'):
        original = json.loads(Path(config['incumbent_config']).read_text(encoding='utf-8'))
        index = int(config.get('incumbent_variant_index', 0))
        settings = dict(original['variants'][index])
    else:
        settings = dict(config.get('bank_settings', config.get('mixed_bank_settings', {})))
    for key in ('method', 'direction_book_path', 'metadata_factor_path', 'budget_bits', 'v_group_size', 'key_metric_path'):
        if key in config:
            settings[key] = config[key]
    settings.setdefault('budget_bits', 1.8)
    settings.setdefault('v_group_size', 'all')
    if settings.get('interaction_policy_path'):
        raise ValueError('Round7 v4 starts from an explicit first-stage priority, not a mixed interaction policy')
    explicit = settings.get('selected_stages')
    if explicit is not None:
        if any(len(value) != 4 or int(value[3]) != 0 for value in explicit):
            raise ValueError('This finite exchange experiment supports first stages only')
        prefix = [unit(value[:3]) for value in explicit]
        if len(set(prefix)) != len(prefix):
            raise ValueError('Duplicate explicit first stages')
        universe = [unit(value) for value in settings.get('selected_units', prefix)]
        if any(value not in universe for value in prefix):
            raise ValueError('Explicit first stages must belong to the calibration ranking universe')
        priority = prefix + [value for value in universe if value not in set(prefix)]
        settings['selected_units'] = [list(value) for value in priority]
        settings['selected_stages'] = stages(priority)
    fixed_count = config.get('fixed_stage_count', settings.get('res_stages_limit',
        settings.get('res_units_limit', settings.get('residual_units_limit'))))
    if fixed_count is not None:
        if isinstance(fixed_count, bool) or int(fixed_count) < 1:
            raise ValueError('A fixed stage count must be a positive integer')
        settings['res_stages_limit'] = int(fixed_count)
    if settings.get('method') == 'round7_key':
        # The actual explicit 224-entry layout replaces the legacy reference
        # allocator. Original config/hash remains in the input manifest.
        settings['residual_layout_reference'] = 'actual'
    return settings


def selected_calibration(config):
    selected, fingerprints = [], {}
    for cohort in config['calibration_sets']:
        samples, references = Path(cohort['samples']), Path(cohort['references'])
        rows = json.loads(samples.read_text(encoding='utf-8-sig'))
        refs = load_references(references)
        chosen = choose_rows(rows, int(cohort.get('source_count', config.get('sources_per_dataset', 2))),
                             int(cohort.get('questions_per_source', config.get('questions_per_source', 2))))
        for row in chosen:
            if str(row['id']) not in refs:
                raise ValueError('Missing calibration reference: ' + str(row['id']))
            selected.append({'dataset': str(cohort['dataset']), 'source_id': source_id(row),
                             'row': row, 'reference_answer': answer(refs[str(row['id'])])})
        fingerprints[str(samples)] = sha(samples)
        fingerprints[str(references)] = sha(references)
    if not selected or len({(r['dataset'], str(r['row']['id'])) for r in selected}) != len(selected):
        raise ValueError('Selected calibration case IDs must be nonempty and unique within datasets')
    return selected, fingerprints


def scores_from_logits(logits, targets, competitors=None):
    import torch
    import torch.nn.functional as F
    values = logits.float()
    nll = F.cross_entropy(values.reshape(-1, values.shape[-1]), targets.reshape(-1), reduction='none')
    if not bool(torch.isfinite(nll).all()):
        raise RuntimeError('Nonfinite packed teacher-forced NLL')
    selected = values.gather(-1, targets[..., None]).squeeze(-1)
    top_values, top_ids = values.topk(2, dim=-1)
    strongest_other = torch.where(top_ids[..., 0] == targets, top_values[..., 1], top_values[..., 0])
    reference_margin = selected - strongest_other
    top_margin = top_values[..., 0] - top_values[..., 1]
    log_probabilities = F.log_softmax(values, dim=-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
    if competitors is None:
        competing = values.clone()
        competing.scatter_(-1, targets[..., None], -torch.inf)
        competitors = competing.argmax(-1)
    margins = selected - values.gather(-1, competitors[..., None]).squeeze(-1)
    return {'mean_nll': float(nll.mean()), 'sum_nll': float(nll.sum()),
            'token_nll': nll.cpu().tolist(), 'token_count': int(targets.numel()),
            'fixed_full_competitor_margin': margins.reshape(-1).cpu().tolist(),
            'reference_margin_target_minus_strongest_other': reference_margin.reshape(-1).cpu().tolist(),
            'top1_top2_margin': top_margin.reshape(-1).cpu().tolist(),
            'entropy_nats': entropy.reshape(-1).cpu().tolist(),
            'first_reference_margin_target_minus_strongest_other': float(reference_margin.reshape(-1)[0]),
            'first_top1_top2_margin': float(top_margin.reshape(-1)[0]),
            'first_entropy_nats': float(entropy.reshape(-1)[0]),
            'first_top1_token_id': int(top_ids[..., 0].reshape(-1)[0]),
            'argmax_token_ids': top_ids[..., 0].reshape(-1).cpu().tolist(),
            'teacher_forced_argmax_token_ids': top_ids[..., 0].reshape(-1).cpu().tolist(),
            'argmax_scope': 'teacher-forced fixed target history; these are not free-running answers',
            'continuous_diagnostics_used_for_selection': False}, competitors


def make_replay(case, target, device):
    import torch
    suffix = case['prompt_suffix'].to(device)
    targets = torch.tensor([target], device=device, dtype=torch.long)
    if len(target) > 1:
        suffix = torch.cat((suffix, targets[:, :-1]), -1)
    return suffix, targets, torch.arange(case['boundary'], case['boundary'] + suffix.shape[-1], device=device)[None]


def prepare_case(model, processor, selected, arguments, config, attention_bridge, stop_ids):
    import torch
    import session_boundary
    from calibrate_gradient_rank_round5 import make_prefix_cache
    row = selected['row']
    feature = {name: value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
               for name, value in session_boundary.features(processor, row, arguments.get('max_patches', 8)).items()}
    visual = (feature['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
    if not visual.numel():
        raise ValueError('No image tokens in calibration case')
    boundary = int(visual[-1]) + 1
    suffix = feature['input_ids'][:, boundary:].clone()
    if not suffix.numel():
        raise ValueError('Compression boundary must precede the entire question suffix')
    with attention_bridge(model, native=True):
        output = model(**feature, use_cache=True, return_dict=True, logits_to_keep=1)
    cache, teacher_ids = output.past_key_values, []
    target_kinds = config.get('target_kinds', ['gt', 'self'])
    if 'self' in target_kinds:
        limit = int(config.get('self_target_tokens', 4))
        if limit < 1:
            raise ValueError('Positive Full self-target horizon required')
        prompt_length = feature['input_ids'].shape[-1]
        for step in range(limit):
            token = int(output.logits[:, -1].argmax(-1))
            teacher_ids.append(token)
            if token in stop_ids or step + 1 == limit:
                break
            with attention_bridge(model, native=True):
                output = model(input_ids=torch.tensor([[token]], device=model.device),
                    position_ids=torch.tensor([[prompt_length + step]], device=model.device),
                    past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
    prefix = [(layer.keys[:, :, :boundary].detach().cpu().clone(),
               layer.values[:, :, :boundary].detach().cpu().clone()) for layer in cache.layers]
    case = {'id': str(row['id']), 'dataset': selected['dataset'], 'source_id': selected['source_id'],
            'visual': visual.cpu(), 'boundary': boundary, 'prompt_suffix': suffix.cpu(), 'prefix': prefix,
            'targets': {}, 'competitors': {}, 'full_scores': {},
            'question_id': row.get('source_question_id', row.get('calibration_question_index'))}
    if 'gt' in target_kinds:
        prefix_text = str(config.get('gt_reference_prefix', ''))
        gt_ids = processor.tokenizer.encode(prefix_text + selected['reference_answer'], add_special_tokens=False)
        eos = processor.tokenizer.eos_token_id
        if eos is None or isinstance(eos, (tuple, list)):
            raise ValueError('One canonical tokenizer EOS token is required for GT targets')
        if not gt_ids or gt_ids[-1] != int(eos):
            gt_ids.append(int(eos))
        if len(gt_ids) > int(config.get('max_gt_tokens', 32)):
            raise ValueError('Fixed GT answer including EOS exceeds max_gt_tokens; select/freeze cases by reference length before launch')
        case['targets']['gt'] = gt_ids
    if 'self' in target_kinds:
        case['targets']['self'] = teacher_ids
    del cache, output, feature
    for name, tokens in case['targets'].items():
        suffix, targets, positions = make_replay(case, tokens, model.device)
        entries = [(k.to(model.device), v.to(model.device)) for k, v in prefix]
        native_cache, _, _ = make_prefix_cache(entries, model.config.get_text_config(), False)
        with attention_bridge(model, native=True):
            output = model(input_ids=suffix, position_ids=positions, past_key_values=native_cache,
                           use_cache=True, return_dict=True, logits_to_keep=len(tokens))
        score, competitor = scores_from_logits(output.logits, targets)
        score.update(target_kind=name, gt_label_conditioned=name == 'gt',
            diagnostic_scope='Full native cached-prefix teacher-forced calibration; source grouping in case record')
        case['full_scores'][name], case['competitors'][name] = score, competitor.cpu()
        del native_cache, output, entries
    return case


class PackedEvaluator:
    def __init__(self, model, cases, settings, base, arguments, output, config):
        self.model, self.cases, self.settings, self.base = model, cases, settings, base
        self.arguments, self.output, self.config = arguments, output, config
        text = model.config.get_text_config()
        self.layers, self.heads = text.num_hidden_layers, text.num_key_value_heads
        self.cache, self.physical_target_forwards, self.physical_allocation_case_evaluations = {}, 0, 0
        self.raw = (output / 'packed_scores.jsonl').open('w', encoding='utf-8')

    def close(self):
        self.raw.close()

    def evaluate(self, ranking, operation='baseline', pair=None, reason=None):
        import torch
        from calibrate_gradient_rank_round5 import make_prefix_cache
        if self.settings.get('method', 'round6_depth') == 'round7_key':
            from round7_key_bank import KeyBank as BankClass
        else:
            from round6_depth_bank import DepthResidualBank as BankClass
        from p184_working_set import attention
        descriptor = {'ranking': [list(value) for value in ranking], 'operation': operation, 'pair': pair}
        key = json.dumps(descriptor, sort_keys=True, separators=(',', ':'))
        if key in self.cache:
            return self.cache[key]
        allocation_id = 'allocation_%03d' % len(self.cache)
        records = []
        for case in self.cases:
            started = time.perf_counter()
            settings, planned = intervention_settings(self.settings, self.base, self.layers, self.heads,
                case['visual'].numel(), ranking, operation, pair)
            record = {'allocation_id': allocation_id, 'id': case['id'], 'dataset': case['dataset'],
                      'source_id': case['source_id'], 'operation': operation, 'pair': pair,
                      'reason': reason, 'scores': {}, 'visual_tokens': int(case['visual'].numel()),
                      'selected_stages': planned['selected_stages'], 'complete_bytes': planned['predicted_bytes'],
                      'allowed_bytes': planned['allowed_bytes'], 'actual_budget_bits': planned['actual_budget_bits'],
                      'ranking_entries': len(settings['selected_stages']), 'ranking_bytes': len(planned['ranking_constant']),
                      'policy_bytes': len(planned['policy_constant']),
                      'bank_method': settings.get('method', 'round6_depth'),
                      'key_index_bytes_per_token': planned['key_index_bytes_per_token'],
                      'book_bytes': planned['book_bytes'],
                      'factor_parameter_bytes': planned['factor_parameter_bytes'],
                      'metric_parameter_bytes': planned['metric_parameter_bytes'],
                      'add_singleton_extra_stage_diagnostic': operation == 'add',
                      'deployment_budget_eligible': operation != 'add', 'fresh_cache_per_target': True}
            for name, tokens in case['targets'].items():
                entries = [(k.to(self.model.device), v.to(self.model.device)) for k, v in case['prefix']]
                cache, _, _ = make_prefix_cache(entries, self.model.config.get_text_config(), False)
                del entries
                BankClass.settings = settings
                bank = None
                try:
                    bank = BankClass(self.model, cache, case['visual'].to(self.model.device), settings.get('method', 'round6_depth'),
                        self.arguments['legacy_scripts'], self.arguments['kivi_source'], keep_reference=False)
                    if any(state.length != case['boundary'] for state in bank.states):
                        raise RuntimeError('A new packed intervention did not start at the clean visual boundary')
                    bank.phase, bank.turn, bank.pass_kind = 'calibration_suffix', 1, 'committed'
                    suffix, targets, positions = make_replay(case, tokens, self.model.device)
                    with attention(self.model):
                        output = self.model(input_ids=suffix, position_ids=positions, past_key_values=cache,
                            use_cache=True, return_dict=True, logits_to_keep=len(tokens))
                    if tuple(output.logits.shape[:2]) != tuple(targets.shape):
                        raise RuntimeError('Packed multi-token teacher-forcing alignment mismatch')
                    score, _ = scores_from_logits(output.logits, targets, case['competitors'][name].to(self.model.device))
                    metadata = bank.result_metadata()
                    if metadata['depth_residual_complete_bytes'] != planned['predicted_bytes']:
                        raise RuntimeError('Planned and actual compact packed owner disagree')
                    if any(reference is not None for reference in bank.references):
                        raise RuntimeError('An intervention bank retained diagnostic FP16 visual references')
                    score.update(encoder_ms=bank.encoding_wall_ms,
                        target_kind=name, gt_label_conditioned=name == 'gt',
                        diagnostic_scope='actual packed cached-prefix teacher-forced calibration; source grouping in allocation-case record',
                        read_calls=sum(value['calls'] for value in bank.event_rows()),
                        owned=bank.owned(), initial_prefix_boundary=case['boundary'],
                        complete_bytes=metadata['depth_residual_complete_bytes'],
                        actual_budget_bits=metadata['depth_residual_actual_budget_bits'])
                    record['scores'][name] = score
                    self.physical_target_forwards += 1
                    del output
                finally:
                    if bank is not None:
                        bank.close()
                    del bank, cache
            self.physical_allocation_case_evaluations += 1
            torch.cuda.synchronize()
            record['seconds'] = time.perf_counter() - started
            records.append(record)
            self.raw.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.raw.flush()
            print(json.dumps({'phase': 'packed_exchange_score', 'allocation': allocation_id,
                'operation': operation, 'pair': None if pair is None else pair['id'], 'id': case['id'],
                'seconds': round(record['seconds'], 2),
                'nll': {name: value['mean_nll'] for name, value in record['scores'].items()}}), flush=True)
        result = {'allocation_id': allocation_id, 'descriptor': descriptor,
                  'scores': {name: macro_scores(records, name) for name in self.cases[0]['targets']},
                  'cases': len(records)}
        self.cache[key] = result
        return result


def evaluate_pairs(evaluator, ranking, pairs, objective, reason):
    baseline = evaluator.evaluate(ranking, reason=reason + ':baseline')
    origin = baseline['scores'][objective]['source_macro_nll']
    rows = []
    for pair in pairs:
        remove = evaluator.evaluate(ranking, 'remove', pair, reason)
        add = evaluator.evaluate(ranking, 'add', pair, reason)
        joint = evaluator.evaluate(ranking, 'joint', pair, reason)
        rows.append({'pair': pair, 'baseline_allocation': baseline['allocation_id'],
            'remove_allocation': remove['allocation_id'], 'add_allocation': add['allocation_id'],
            'joint_allocation': joint['allocation_id'],
            'predicted_improvement_additive': (2 * origin - remove['scores'][objective]['source_macro_nll']
                                               - add['scores'][objective]['source_macro_nll']),
            'actual_improvement_joint': origin - joint['scores'][objective]['source_macro_nll'],
            'per_target': {kind: {
                'predicted_improvement_additive': (2 * baseline['scores'][kind]['source_macro_nll']
                    - remove['scores'][kind]['source_macro_nll'] - add['scores'][kind]['source_macro_nll']),
                'actual_improvement_joint': baseline['scores'][kind]['source_macro_nll'] - joint['scores'][kind]['source_macro_nll']}
                for kind in baseline['scores']}})
    return baseline, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    args.output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(config)
    selected, input_hashes = selected_calibration(config)
    arguments = config['arguments']
    sys.path.insert(0, str(ROOT / 'code/experiments'))
    from bank import install_legacy
    install_legacy(arguments['legacy_scripts'])
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from p180_multiturn_core import attention_bridge
    import run_p175_modern_replication as oldbase
    torch.manual_seed(int(config.get('seed', 1212)))
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    base = json.loads(Path(settings['metadata_factor_path']).read_text())
    if base['encoder_book_sha256'] != sha(settings['direction_book_path']):
        raise ValueError('Frozen base metadata/encoder book mismatch')
    if base['value_group_size'] != settings['v_group_size']:
        raise ValueError('Frozen metadata V-group mismatch')
    input_hashes.update({str(args.config): sha(args.config),
        settings['direction_book_path']: sha(settings['direction_book_path']),
        settings['metadata_factor_path']: sha(settings['metadata_factor_path'])})
    if settings.get('key_metric_path'):
        input_hashes[settings['key_metric_path']] = sha(settings['key_metric_path'])
    gradient = None
    if config.get('gradient_ranking_path'):
        path = config['gradient_ranking_path']
        gradient = json.loads(Path(path).read_text())
        input_hashes[path] = sha(path)
    if config.get('incumbent_config'):
        input_hashes[config['incumbent_config']] = sha(config['incumbent_config'])
    # Other Round7 agents may continue editing independent new operators while
    # this run is active. Freeze this executable and its actual old dependencies.
    source_paths = ('code/a2/round7_exchange_key.py', 'code/a2/round7_exchange_policy.py',
        'code/a2/round6_depth_bank.py', 'code/a2/round6_depth_codec.py',
        'code/a2/round6_depth_attention.py', 'code/a2/masked_bank.py',
        'code/a2/metadata_direction_bank.py', 'code/a2/learned_direction_bank.py',
        'code/a2/affine_rate_bank.py', 'code/a2/quantizers.py', 'code/a2/fast_wht.py',
        'code/a2/fused_masked.py', 'code/a2/fused_learned_direction.py',
        'code/a2/session_boundary.py', 'code/a2/calibrate_gradient_rank_round5.py',
        'code/experiments/fused_progressive.py', 'code/experiments/check_fused_progressive.py',
        'code/experiments/bank.py')
    if settings.get('method', 'round6_depth') == 'round7_key':
        source_paths += ('code/a2/round7_key_bank.py', 'code/a2/round7_key_attention.py',
            'code/a2/round7_key_codec.py', 'code/a2/round7_key_layout.py', 'code/a2/round7_key_artifacts.py')
    sources = {relative: sha(ROOT / relative) for relative in source_paths}
    manifest = {'schema': 'trajkv_round7_packed_conditional_exchange_v4', 'status': 'RUNNING',
        'config': config, 'input_sha256': input_hashes, 'source_sha256': sources,
        'calibration_cases': [{'id': str(item['row']['id']), 'dataset': item['dataset'],
            'source_id': item['source_id']} for item in selected],
        'gold_answers_loaded': 'gt' in config.get('target_kinds', ['gt', 'self']),
        'GT_rule': 'fixed first reference; fixed configured reference prefix; canonical EOS included; no answer-oracle choice',
        'model_weights_frozen': True, 'source_independent_sample_unit': 'document/image source, not intervention x question',
        'quality_scope': 'cached-prefix calibration diagnostic; no end-to-end accuracy or timing claim',
        'reader': ('round7_key_bank + round7_key_attention real nibble/uint8 integer codes and compact FP16-alpha stages'
            if settings.get('method') == 'round7_key' else 'unmodified round6_depth_bank + round6_depth_attention real compact integer codes/FP16 alpha'),
        'fixed_stage_count': settings.get('res_stages_limit'),
        'reset_scope': 'fresh DynamicCache and freshly encoded packed bank for every allocation x case x target',
        'singleton_scope': 'remove k-1 with no cold replacement; add k+1 with one diagnostic extra stage; joint/final <= fixed deployment budget',
        'budget_scope': 'complete 224-entry stage0 priority, uint16 448B, unchanged 5B explicit descriptor and full lookup',
        'selection_objective': config.get('selection_objective', 'gt'),
        'continuous_diagnostics': 'per-token reference margin vs current strongest other, top1-top2 margin, entropy nats, teacher-forced argmax IDs; never a selection gate',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    write_json(args.output / 'manifest.json', manifest)
    processor = AutoProcessor.from_pretrained(arguments['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(arguments['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text = model.config.get_text_config()
    ranking = ranking_units(settings, text.num_hidden_layers, text.num_key_value_heads)
    cases = []
    with torch.inference_mode():
        for item in selected:
            case = prepare_case(model, processor, item, arguments, config, attention_bridge, oldbase.stop_ids(model, processor))
            cases.append(case)
            serial = {key: value for key, value in case.items() if key not in ('prefix', 'prompt_suffix', 'visual', 'competitors')}
            serial.update(visual_tokens=int(case['visual'].numel()), prompt_suffix_tokens=case['prompt_suffix'].shape[-1],
                          fixed_full_competitor_token_ids={kind: value.reshape(-1).tolist() for kind, value in case['competitors'].items()},
                          prefix_diagnostic_fp16_bytes=sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in case['prefix']))
            with (args.output / 'cases.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(serial, ensure_ascii=False) + '\n')
            print(json.dumps({'phase': 'prepare_full_targets', 'id': case['id'], 'dataset': case['dataset'],
                'target_tokens': {name: len(value) for name, value in case['targets'].items()}}), flush=True)
        plans = [depth_plan(settings, base, text.num_hidden_layers, text.num_key_value_heads,
                            case['visual'].numel(), ranking) for case in cases]
        active = [{tuple(value[:3]) for value in plan['selected_stages']} for plan in plans]
        pairs = candidates(config, ranking, active, gradient)
        batches = int(config.get('batches', 2))
        if not 1 <= batches <= len(pairs):
            raise ValueError('batches must be between one and the frozen candidate count')
        partitions = [pairs[index * len(pairs) // batches:(index + 1) * len(pairs) // batches] for index in range(batches)]
        write_json(args.output / 'frozen_candidates.json', {'pairs': pairs, 'batches': partitions,
            'initial_selected_by_case': [{'id': case['id'], 'stages': plan['selected_stages']} for case, plan in zip(cases, plans)],
            'candidate_selection': 'fixed full-native gradient rank: lowest active removals / highest inactive additions; never fitted to current NLL',
            'no_new_threshold_gate': True})
        evaluator = PackedEvaluator(model, cases, settings, base, arguments, args.output, config)
        objective = config.get('selection_objective', 'gt')
        if objective not in cases[0]['targets']:
            raise ValueError('selection_objective must have a collected fixed target sequence')
        traces, static_rows, current, accepted = [], [], list(ranking), []
        try:
            initial = evaluator.evaluate(ranking, reason='initial_incumbent_explicit_stage0')
            for index, batch in enumerate(partitions):
                _, static = evaluate_pairs(evaluator, ranking, batch, objective, 'static_batch_%d' % index)
                static_rows.extend(static)
                _, conditional = evaluate_pairs(evaluator, current, batch, objective, 'conditional_batch_%d' % index)
                chosen = pick_disjoint(conditional, 'actual_improvement_joint', 1)
                before = current
                if chosen:
                    current = apply_swaps(current, chosen)
                    accepted.extend(chosen)
                after = evaluator.evaluate(current, reason='accepted_student_recompute_batch_%d' % index)
                # Give all three methods the same post-batch validation slot.
                # Static scores always remain anchored to the initial student;
                # these verified partial policies never feed back into ranking.
                static_checks = {}
                for label, score_key in (('one_shot_additive', 'predicted_improvement_additive'),
                                         ('static_whole_pair', 'actual_improvement_joint')):
                    selected_pairs = pick_disjoint(static_rows, score_key, index + 1)
                    verified = evaluator.evaluate(apply_swaps(ranking, selected_pairs),
                        reason='static_partial_policy:%s:batch_%d' % (label, index))
                    static_checks[label] = {'selected_pairs': selected_pairs,
                        'allocation_id': verified['allocation_id'], 'scores': verified['scores'],
                        'validation_does_not_update_static_candidate_scores': True}
                trace = {'batch': index, 'candidate_ids': [pair['id'] for pair in batch],
                    'static_initial_state_scores': static, 'conditional_current_state_scores': conditional,
                    'accepted': chosen, 'student_state_changed': before != current,
                    'after_acceptance_allocation': after['allocation_id'], 'after_scores': after['scores'],
                    'static_partial_policy_verification': static_checks}
                traces.append(trace)
                write_json(args.output / 'trace.json', traces)
            additive_pairs = pick_disjoint(static_rows, 'predicted_improvement_additive', batches)
            joint_pairs = pick_disjoint(static_rows, 'actual_improvement_joint', batches)
            policies = {'incumbent': ranking, 'one_shot_additive': apply_swaps(ranking, additive_pairs),
                        'static_whole_pair': apply_swaps(ranking, joint_pairs), 'conditional_batches': current}
            results = {name: evaluator.evaluate(value, reason='final_policy:' + name) for name, value in policies.items()}
            deployments = {}
            for name, priority in policies.items():
                variant = dict(settings)
                variant.pop('selected_units', None)
                variant.pop('depth_policy', None)
                variant.update(method=settings.get('method', 'round6_depth'), name='r7_exchange_' + name, selected_stages=stages(priority))
                for key in ('res_units_limit', 'residual_units_limit', 'res_stages_limit'):
                    variant.pop(key, None)
                if settings.get('res_stages_limit') is not None:
                    variant['res_stages_limit'] = int(settings['res_stages_limit'])
                deployments[name] = {'schema': 'trajkv_round7_full_stage0_priority_v1',
                    'variant': variant, 'selected_units_priority': [list(item) for item in priority],
                    'ranking_entries': len(priority), 'ranking_constant_bytes': len(priority) * 2,
                    'all_stages_are_first_stage': True, 'parent_before_child': True,
                    'fixed_stage_count': settings.get('res_stages_limit'),
                    'deployment_scope': 'global first-stage ranking; affordable prefix selected by actual visual N and complete owner',
                    'calibration_score': results[name]['scores']}
                write_json(args.output / ('policy_' + name + '.json'), deployments[name])
            logical = 2 * batches + 3 * len(pairs) + 1
            report = {'schema': 'trajkv_round7_packed_conditional_exchange_result_v4', 'status': 'COMPLETE',
                'bank_method': settings.get('method', 'round6_depth'),
                'fixed_stage_count': settings.get('res_stages_limit'),
                'selection_objective': objective, 'initial': initial, 'final': results,
                'accepted_pairs': {'one_shot_additive': additive_pairs, 'static_whole_pair': joint_pairs,
                                   'conditional_batches': accepted},
                'static_pair_scores': static_rows, 'trace': traces,
                'same_frozen_candidate_pairs': True,
                'logical_allocation_evaluations_per_method': logical,
                'logical_budget_formula': 'one anchor and one resulting-policy verification per batch + remove/add/joint per frozen pair + final-policy verification',
                'logical_target_forwards_per_method': logical * sum(len(case['targets']) for case in cases),
                'shared_exact_input_results_reused': True,
                'physical_allocation_case_evaluations': evaluator.physical_allocation_case_evaluations,
                'physical_target_forwards': evaluator.physical_target_forwards,
                'unique_allocations': len(evaluator.cache),
                'independent_sources': len({(case['dataset'], case['source_id']) for case in cases}),
                'questions': len(cases), 'no_claim_of_independent_intervention_samples': True,
                'primary_quality_eligible': False, 'end_to_end_timing_eligible': False,
                'policies': list(deployments)}
            write_json(args.output / 'summary.json', report)
            write_json(args.output / 'allocation_summary.json', list(evaluator.cache.values()))
        finally:
            evaluator.close()
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen calibration input changed: ' + path)
    for relative, digest in sources.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Frozen calibration source changed: ' + relative)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Model weights were not frozen throughout calibration')
    manifest['status'] = 'COMPLETE'
    manifest['legacy_source_sha256'] = {str(Path(module.__file__)): sha(module.__file__) for name, module in sys.modules.items()
        if name.startswith(('p176_', 'p180_', 'p184_', 'p185_', 'p201_', 'p202_')) and getattr(module, '__file__', None)
        and Path(module.__file__).is_file()}
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'complete.json', {'status': 'COMPLETE', 'questions': len(cases),
        'independent_sources': report['independent_sources'], 'physical_target_forwards': report['physical_target_forwards']})
    print(json.dumps({'status': 'COMPLETE', 'questions': len(cases), 'sources': report['independent_sources'],
        'physical_target_forwards': report['physical_target_forwards'],
        'final_objective_nll': {name: value['scores'][objective]['source_macro_nll'] for name, value in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
