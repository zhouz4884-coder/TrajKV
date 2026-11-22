"""KRA1: reuse frozen fit48 query states, then actual-model select16.

No new fit-state capture, reference answers, task quality scores or GPU queue
operations. Old calibration functions are explicitly reused only for native
prefix preparation, teacher EOS/cap32 and packed full-suffix TF replay.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round10_calibrate_v1 as common
import round10_runner_v1 as driver

SCHEMA = 'trajkv.round10.kra_calibration_config.v1'
PARENT_COMPLETE_SHA = '80baf58e70b350aa26dbc4737c4295bb7003a36ed5dc9b4769af9bdc84dc2405'
OLD_CAL_SHA = 'abbc45f07e5899ff6018e0a22c5bba94c8764b03cd518e764de1949d9df87a8a'
STRENGTHS = (0., .25, .5, 1.)
BASE_DRIVER_FILES = ('round10_runner_v1.py', 'round10_value_bank_v2.py', 'round10_value_bank_v1.py',
    'round9_value_bank_v1.py', 'r7_legacy_bridge_v1.py', 'r7_legacy_bridge_v2.py', 'm1_protocol.py',
    'model_registry.py', 'dataset_registry.py', 'evaluate.py', 'cache_bridge.py', 'attention_bridge.py',
    'ownership.py', 'round10_inputs_v1.py')
FILENAMES = {.25: 'metric_strength025.json', .5: 'metric_strength05.json', 1.: 'metric_strength1.json'}
require, sha, canonical, write_new = common.require, common.sha, common.canonical, common.write_new
tensor_sha = common.tensor_sha


def validate_config(config):
    require(config.get('schema') == SCHEMA and config.get('purpose') == 'calibration_only',
            'explicit new KRA calibration schema/scope')
    require(not common.FORBIDDEN.intersection(config), 'no labels or references')
    require(config.get('model_id') == 'M1' and config.get('seed') == 0 and
            config.get('question_suffix') == common.SUFFIX and
            config.get('preprocessing', {}).get('max_patches') == 8 and
            config.get('teacher_max_new_tokens', 32) == 32,
            'frozen M1 seed0 cap8 question suffix and teacher EOS/cap32')
    require(config.get('variant_source', {}).get('name') == common.BASE_PROFILE and
            config['variant_source'].get('sha256') == driver.ORIGINAL_CONFIG_SHA and
            config.get('expected_asset_sha256') == driver.PROFILES[common.BASE_PROFILE]['assets'],
            'original low43 source and assets only')
    for role in ('fit', 'select'):
        require(isinstance(config.get(f'cal_{role}_inputs'), str) and
                len(config.get(f'cal_{role}_sha256', '')) == 64, 'locked ' + role + ' rows')
    require(isinstance(config.get('closed_collection_dir'), str) and
            config.get('closed_collection_complete_sha256') == PARENT_COMPLETE_SHA,
            'original closed cal64 collection identity')
    expected = config.get('expected_source_hashes')
    require(isinstance(expected, dict) and
            {'code/round10_kra_calibrate_v1.py', 'code/round10_kra_bank_v1.py',
             'code/round10_kra_metric_v1.py'}.issubset(expected), 'explicit new source graph')
    for key, digest in expected.items():
        require(key.startswith('code/') and '..' not in Path(key).parts and len(digest) == 64,
                'safe project source identity')
    require(type(config.get('technical_only', False)) is bool, 'technical_only bool')
    return config


def source_manifest():
    files = set(common.PINS) | {'code/' + name for name in BASE_DRIVER_FILES} | {
        'code/round10_calibrate_v1.py', 'code/round10_kra_calibrate_v1.py',
        'code/round10_kra_bank_v1.py', 'code/round10_kra_metric_v1.py'}
    return {key: sha(ROOT / key) for key in sorted(files)}


def check_sources(config):
    result = source_manifest()
    require(result['code/round10_calibrate_v1.py'] == OLD_CAL_SHA and
            result['code/round10_runner_v1.py'] ==
            '22b5e45152ea9be982ae7ba77950ed09bd065ae574b58b80725beb0983c54e96',
            'frozen calibration helpers or generation contract changed')
    for locks in (common.PINS, config['expected_source_hashes']):
        for key, digest in locks.items():
            require(result.get(key) == digest, 'source lock differs: ' + key)
    return result


def scaled_metric(document, strength, parent_sha):
    import torch
    require(strength in STRENGTHS, 'finite strength grid')
    result = dict(document)
    result.update(gamma=(torch.tensor(document['gamma'], dtype=torch.float32) * strength).tolist(),
                  strength=strength, parent_metric_sha256=parent_sha)
    return result


def select_strength(losses, expected_sources):
    require(set(losses) == set(STRENGTHS), 'include zero control and all three nonzero strengths')
    means = {}
    for strength, rows in losses.items():
        require(set(rows) == set(expected_sources), 'every source at each strength')
        require(all(math.isfinite(v) and v >= -1e-10 for v in rows.values()), 'finite KL')
        means[strength] = sum(rows.values()) / len(expected_sources)
    selected = min(STRENGTHS[1:], key=lambda value: (means[value], value))
    return selected, means


def immutable_field_signature(owners):
    """Hash everything read by R7 except the original K alpha slots."""
    import torch
    values = []
    for owner in owners:
        state = owner.state
        fields = {key: tensor_sha(value) for key, value in vars(state).items()
                  if isinstance(value, torch.Tensor) and
                  key not in ('text_k', 'text_v', 'text_pos', 'depth_residual_alpha')}
        lookup = state.depth_residual_lookup
        for head in range(4):
            for stage in range(2):
                slot = int(lookup[1, head, stage])
                if slot >= 0:
                    fields[f'V_alpha_{head}_{stage}'] = tensor_sha(state.depth_residual_alpha[slot])
        values.append(fields)
    return canonical(values)


def build_kra_cache(native, batch, variant, config, metric):
    from m1_protocol import clone_m1_prefix_cache
    from cache_bridge import HybridVisualCache
    from r7_frozen_pilot_v2 import share_original_visual_position_map
    from round10_kra_bank_v1 import build_round10_kra_owners
    clone = clone_m1_prefix_cache(native)
    owners, metadata = build_round10_kra_owners(clone, batch.visual_positions, variant,
        metric_override=metric, calibration_only=True,
        **{key: config[key] for key in ('legacy_a2', 'legacy_experiments', 'legacy_scripts', 'kivi_source')},
        expected_asset_sha256=config['expected_asset_sha256'])
    signatures = common.state_signatures(owners)
    signatures['immutable_fields_sha256'] = immutable_field_signature(owners)
    cache = HybridVisualCache(clone, batch.visual_positions, owners, consume_native=True)
    share_original_visual_position_map(cache)
    return cache, owners, metadata, signatures


def replay_with_logits_identity(model, batch, cache, owners, history):
    observed = []
    def capture(module, args, kwargs, result):
        observed.append(tensor_sha(result.logits[0, -1].detach().float().cpu()))
    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        result = common.replay_branch(model, batch, cache, owners, history)
    finally:
        handle.remove()
    require(len(observed) == len(history['token_ids']), 'full suffix logits identity scope')
    return {**result, 'actual_raw_logits_per_position_sha256': observed,
            'actual_raw_logits_sha256': canonical(observed)}


def calibrate(config, output, *, identity_only=False):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from round10_inputs_v1 import load_round10_inputs
    from round10_kra_metric_v1 import fit_metric, load_metric
    from r7_frozen_pilot_v2 import frozen_variant
    from attention_bridge import language_attention_modules
    from ownership import cache_storage
    from model_registry import get_model_spec
    from evaluate import model_identity
    validate_config(config)
    source_before = check_sources(config)
    identity_only = identity_only or config.get('technical_only', False)
    rows = {role: load_round10_inputs(config[f'cal_{role}_inputs'], config.get('input_manifest_path'),
        expected_sha256=config[f'cal_{role}_sha256'],
        expected_role=config.get(f'cal_{role}_role', f'cal_{role}')) for role in ('fit', 'select')}
    common.validate_rows(rows['fit'], rows['select'])
    collection = Path(config['closed_collection_dir'])
    require(collection.name == 'cal64_v1' and
            sha(collection / 'complete.json') == PARENT_COMPLETE_SHA, 'frozen parent collection SHA')
    parent_manifest = json.loads((collection / 'manifest.json').read_text(encoding='utf-8-sig'))
    for role in ('fit', 'select'):
        require(parent_manifest['config'][f'cal_{role}_sha256'] == config[f'cal_{role}_sha256'],
                'same original calibration row identities')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base = fit_metric(collection, output / 'metric_fit48.json',
                      expected_parent_complete_sha256=PARENT_COMPLETE_SHA)
    base_sha = sha(output / 'metric_fit48.json')
    grid = {strength: scaled_metric(base, strength, base_sha) for strength in STRENGTHS}
    # The zero-strength JSON is an audit control, never the selected candidate.
    paths = {0.: output / 'metric_strength0_control.json',
             **{strength: output / filename for strength, filename in FILENAMES.items()}}
    metrics = {}
    for strength, path in paths.items():
        write_new(path, grid[strength])
        metrics[strength], _ = load_metric(path, sha(path), device='cpu')
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6),
            'actual CUDA SM86 required')
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variant = frozen_variant(config)
    require(sha(variant['metadata_factor_path']) == common.BASE_FACTOR_SHA, 'original V0 factor SHA')
    identity = model_identity(config, get_model_spec('M1'))
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
        device_map={'': 0}, attn_implementation='sdpa', local_files_only=True).eval()
    model.requires_grad_(False)
    require(all(p.dtype == torch.float16 for p in model.parameters()), 'native FP16 unquantized weights')
    require(int(model.generation_config.num_beams or 1) == 1, 'native greedy beam-one')
    _, modules = language_attention_modules(model)
    require(len(modules) == 28, 'M1 all28 geometry')
    manifest = {'schema': 'trajkv.round10.kra_calibration_run.v1', 'config': config,
        'runtime_pid': os.getpid(), 'runtime_executable': sys.executable,
        'runtime_argv': [sys.executable, *sys.argv], 'runtime_source_root': str(ROOT),
        'source_hashes': source_before, 'model_identity': identity,
        'fit_sources_reused': 48, 'new_fit_source_captures': 0,
        'fit_method': 'uncentered source-balanced actual compressed q_rotated PCA rank1; not LS48',
        'parent_calibration_complete_sha256': PARENT_COMPLETE_SHA,
        'identity_only': identity_only, 'expected_select_sources': 1 if identity_only else 16,
        'teacher_generation': {'max_new_tokens': 32, 'native_generation_config': model.generation_config.to_dict()},
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'python_version': platform.python_version(), 'actual_GPU': torch.cuda.get_device_name(),
        'weight_dtype': 'float16', 'tf32': False, 'references_read': False,
        'question_unknown_encoding': True, 'metric_storage_bytes': 5980,
        'private_parent_fit_shards_not_public': True,
        'actual_processor_class': type(processor).__module__ + '.' + type(processor).__name__,
        'actual_tokenizer_class': type(processor.tokenizer).__module__ + '.' + type(processor.tokenizer).__name__,
        'teacher_loss_arithmetic': 'CPU float64 full vocabulary from actual GPU logits',
        'reused_frozen_helpers': ['prepare_case', 'teacher_history', 'replay_branch', 'full_vocab_kl'],
        'deployment_peak_or_speed_benchmark': False}
    write_new(output / 'manifest.json', manifest)
    losses = {strength: {} for strength in STRENGTHS}
    counts = Counter()
    identity_receipt = None
    records = (output / 'records.jsonl').open('x', encoding='utf-8', newline='\n')
    def log(value):
        records.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        records.flush()
        print(json.dumps({key: value[key] for key in ('phase', 'index', 'id', 'strength') if key in value}), flush=True)
    try:
        selection_rows = rows['select'][:1] if identity_only else rows['select']
        for index, row in enumerate(selection_rows):
            batch, native, context = common.prepare_case(model, processor, row)
            history = common.teacher_history(model, batch, native)
            counts['teacher_generated_sources'] += 1
            counts['teacher_generated_tokens'] += len(history['token_ids'])
            source = row['dataset'] + ':' + row['source_id']
            log({'phase': 'select_source_teacher', 'index': index, 'id': row['id'],
                 'dataset': row['dataset'], 'source_id': row['source_id'], 'context': context,
                 'teacher_token_ids': history['token_ids'], 'teacher_logits_sha256': tensor_sha(history['raw_logits']),
                 'first_call': history['first_call'], 'EOS_reached': history['EOS_reached'],
                 'teacher_cap': 32, 'references_read': False})
            cache, owners, metadata, reference = common.build_cache(native, batch, variant, config)
            immutable = immutable_field_signature(owners)
            original_bytes = cache_storage(cache, cache.visual_positions)['stored_visual_KV_bytes']
            require(original_bytes == 4590*int(batch.visual_positions.numel()) + 89733, 'original B0 formula')
            original_loss = None
            if index == 0:
                original_loss = replay_with_logits_identity(model, batch, cache, owners, history)
                counts['original_identity_TF_source_evaluations_actual'] += 1
                counts['TF_student_outer_forwards_actual'] += original_loss['positions']
            cache = owners = metadata = None
            strengths = (0.,) if identity_only else STRENGTHS
            for strength in strengths:
                cache, owners, metadata, signature = build_kra_cache(native, batch, variant, config, metrics[strength])
                require(signature['immutable_fields_sha256'] == immutable,
                        'candidate changed V/all codes/Kbase/lookup/positions')
                if strength == 0:
                    require(signature['reader_field_sha256'] == reference['reader_field_sha256'],
                            't0 original complete reader fields differ')
                owner = cache_storage(cache, cache.visual_positions)
                complete_bytes = owner['stored_visual_KV_bytes']
                require(complete_bytes == original_bytes + 5980 and
                        metadata['round10_kra_candidate']['new_persistent_constant_bytes'] == 5980,
                        'actual metric shared storage not fully charged')
                scores = replay_with_logits_identity(model, batch, cache, owners, history)
                counts['TF_source_evaluations_actual'] += 1
                counts['TF_student_outer_forwards_actual'] += scores['positions']
                if strength == 0 and index == 0:
                    require(scores['actual_raw_logits_sha256'] == original_loss['actual_raw_logits_sha256'] and
                            scores['position_KL'] == original_loss['position_KL'],
                            't0 actual full model logits differ from original R7')
                    identity_receipt = {'schema': 'trajkv.round10.kra_identity.v1',
                        'status': 'PASS_T0_FIELDS_AND_ACTUAL_MODEL_LOGITS_IDENTICAL',
                        'actual_GPU': True, 'context': context, 'teacher_token_ids': history['token_ids'],
                        'reader_field_sha256': signature['reader_field_sha256'],
                        'immutable_fields_sha256': immutable,
                        'original_R7_actual_TF': original_loss, 'KRA1_t0_actual_TF': scores,
                        'original_complete_bytes': original_bytes,
                        'KRA1_t0_complete_bytes': complete_bytes, 'actual_metric_increment_bytes': 5980,
                        'new_free_generation_task_answers': 0, 'references_read': False}
                    write_new(output / 'identity.json', identity_receipt)
                losses[strength][source] = scores['source_mean_KL']
                log({'phase': 'select_branch', 'index': index, 'id': row['id'],
                     'dataset': row['dataset'], 'source_id': row['source_id'], 'context': context,
                     'mode': 'KRA1', 'strength': strength, 'metric_file': str(paths[strength]),
                     'metric_sha256': sha(paths[strength]), 'loss': scores,
                     'reader_field_sha256': signature['reader_field_sha256'],
                     'immutable_fields_sha256': immutable,
                     'owner_before_question_suffix': owner, 'owner_metadata': metadata,
                     'calibration_keeps_native_reference': True,
                     'not_deployment_peak_or_quality_measurement': True})
                cache = owners = metadata = signature = None
                gc.collect()
            batch = native = history = reference = None
            torch.cuda.empty_cache()
        require(identity_receipt is not None, 'actual t0 identity receipt required')
        selected = None
        selection = None
        if not identity_only:
            expected_sources = [row['dataset'] + ':' + row['source_id'] for row in rows['select']]
            selected, means = select_strength(losses, expected_sources)
            selection = {'schema': 'trajkv.round10.kra_selection.v1', 'selected_strength': selected,
                'strength_grid': list(STRENGTHS), 'source_balanced_mean_KL': {str(k): v for k, v in means.items()},
                'selection_objective': 'source-balanced actual packed full-suffix full-vocabulary teacher-forced KL',
                'selected_on_calSelect16_only': True, 'zero_is_control_not_candidate_gate': True,
                'nonzero_candidate_is_research_test_not_quality_certification': True,
                'cal_fit_inputs_sha256': config['cal_fit_sha256'], 'cal_select_inputs_sha256': config['cal_select_sha256'],
                'teacher_EOS_cap32': True, 'references_read': False,
                'parent_calibration_complete_sha256': PARENT_COMPLETE_SHA,
                'source_hashes': source_before, 'metric_storage_bytes': 5980}
            write_new(output / 'selection.json', selection)
            selected_doc = {**grid[selected], 'selection_provenance': selection}
            write_new(output / 'selected_metric.json', selected_doc)
            load_metric(output / 'selected_metric.json', sha(output / 'selected_metric.json'), device='cpu')
        require(check_sources(config) == source_before, 'source changed during calibration')
        result = {'schema': 'trajkv.round10.kra_calibration_complete.v1',
            'status': ('PASS_KRA_T0_ACTUAL_MODEL_IDENTITY' if identity_only else
                       'PASS_KRA_FIT48_REUSED_AND_ACTUAL_MODEL_SELECT16'),
            'actual_GPU': True, 'fit_sources_reused': 48, 'new_fit_source_captures': 0,
            'select_sources': len(selection_rows), 'teacher_cap': 32, 'selected_strength': selected,
            'metric_file': None if identity_only else str(output / 'selected_metric.json'),
            'metric_sha256': None if identity_only else sha(output / 'selected_metric.json'),
            'selection_sha256': None if identity_only else sha(output / 'selection.json'),
            'identity_sha256': sha(output / 'identity.json'), 'source_hashes': source_before,
            'parent_calibration_complete_sha256': PARENT_COMPLETE_SHA,
            'records_sha256': sha(output / 'records.jsonl'), 'counts': dict(counts),
            'references_read': False, 'new_free_generation_task_answers': 0,
            'teacher_generation_is_separately_counted': True,
            'quality_or_improved_incumbent_claim': False, 'wall_s': time.perf_counter() - started}
        write_new(output / 'complete.json', result)
        return result
    except Exception as error:
        write_new(output / 'failure.json', {'schema': 'trajkv.round10.kra_calibration_failure.v1',
            'status': 'FAILED_ACTUAL_KRA_CALIBRATION_NO_SYNTHETIC_COMPLETION',
            'error_type': type(error).__name__, 'error': str(error), 'counts': dict(counts),
            'references_read': False, 'new_free_generation_task_answers': 0})
        raise
    finally:
        records.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    config['calibration_config_sha256'] = sha(args.config)
    result = calibrate(config, args.output, identity_only=args.identity_only)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
