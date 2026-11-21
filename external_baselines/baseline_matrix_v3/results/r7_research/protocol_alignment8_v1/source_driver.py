"""Real M1 FP16 protocol alignment; four paths, no codec or task scoring.

A: independent native complete-prompt forward and native greedy generation.
B: image-prefix-only prefill, then the common complete question suffix.
C: native complete-prompt prefill, crop to the image boundary, same suffix.
D: image-prefix-only prefill -> FP16 HybridVisualCache, same suffix.

Suffix-logit measurement and generation ALWAYS receive separate cache clones.
Native prefix prototypes retained here are diagnostic storage, not a low-bit
owner or a speed/memory benchmark. Numerical differences alone are observations.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
SCHEMA = 'trajkv.r7_research.protocol_alignment.v1'
FORBIDDEN = {'refs', 'references', 'reference_answers', 'reference_path', 'answers', 'labels', 'scores'}
SUFFIX = '\nAnswer the question using a single word or phrase.'
ARMS = ('native_full', 'prefix_only', 'full_prefill_cropped', 'prefix_fp16_hybrid')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(config):
    require(config.get('schema') == SCHEMA, 'SCHEMA: exact diagnostic schema required')
    require(not (set(config) & FORBIDDEN), 'SCHEMA: reference/score fields forbidden')
    require(config.get('model_id') == 'M1', 'IDENTITY: M1 only')
    require(config.get('seed', 0) == 0, 'IDENTITY: seed0 required')
    require(config.get('comparison_arms') == list(ARMS), 'SCOPE: exact four declared controls required')
    require(config.get('question_suffix', SUFFIX) == SUFFIX, 'IDENTITY: original question suffix required')
    require(config.get('preprocessing', {}).get('max_patches') == 8, 'IDENTITY: max_patches=8 required')
    require(config.get('max_new_tokens') in (32, 64), 'SCHEMA: answer cap must be32 or64')
    entries = config.get('inputs')
    require(isinstance(entries, list) and len(entries) == 2, 'SCHEMA: two explicit input-file selections required')
    require(all(isinstance(entry, dict) for entry in entries), 'SCHEMA: input selections must be objects')
    require({entry.get('dataset') for entry in entries} == {'textvqa', 'docvqa'}, 'IDENTITY: Text and Doc only')
    for entry in entries:
        require(isinstance(entry, dict) and not (set(entry) & FORBIDDEN), 'SCHEMA: answer-free input descriptor required')
        require(isinstance(entry.get('path'), str) and entry['path'], 'SCHEMA: explicit input path required')
        require(entry.get('start', 0) == 0 and entry.get('count') == 4, 'SCOPE: fixed first4 per dataset')
    return entries


def finite(tensor, stage):
    import torch
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError('NONFINITE: ' + stage + ' nan=' + str(int(torch.isnan(tensor).sum()))
                                 + ' inf=' + str(int(torch.isinf(tensor).sum())))


def tensor_sha(tensor):
    from evaluate import tensor_sha as frozen_tensor_sha
    return frozen_tensor_sha(tensor)


def tensor_contract(tensor, include_ids=False):
    result = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': tensor_sha(tensor)}
    if include_ids:
        result['ids'] = tensor.detach().cpu().reshape(-1).tolist()
    return result


def logits_difference(candidate, reference):
    import torch
    require(candidate.shape == reference.shape, 'SCHEMA: comparable vocabulary shapes required')
    finite(candidate, 'candidate last logits'); finite(reference, 'reference last logits')
    a, b = candidate.double().reshape(-1), reference.double().reshape(-1)
    delta = a - b
    lb, la = torch.log_softmax(b, -1), torch.log_softmax(a, -1)
    return {'max_abs': float(delta.abs().max()), 'mean_abs': float(delta.abs().mean()),
            'relative_L2': float(delta.norm() / b.norm().clamp_min(1e-30)),
            'KL_reference_to_candidate': max(0., float((lb.exp() * (lb-la)).sum())),
            'KL_candidate_to_reference': max(0., float((la.exp() * (la-lb)).sum())),
            'reference_top1': int(b.argmax()), 'candidate_top1': int(a.argmax()),
            'top1_equal': bool(a.argmax() == b.argmax()),
            'difference_is_observation_not_tolerance_gate': True}


def validate_prefix(cache, boundary, spec, stage):
    require(len(cache.layers) == spec['layers'] and int(cache.get_seq_length()) == boundary,
            'SCHEMA: native prefix length/layers differ at ' + stage)
    for index, layer in enumerate(cache.layers):
        for axis, value in (('K', layer.keys), ('V', layer.values)):
            require(list(value.shape) == [1, spec['kv_heads'], boundary, spec['head_dim']],
                    'SCHEMA: prefix geometry differs at ' + stage)
            require(str(value.dtype) == 'torch.float16' and value.is_cuda, 'IDENTITY: FP16 CUDA prefix required')
            finite(value, stage + '/layer' + str(index) + '/' + axis)


def cache_difference(candidate, reference):
    import torch
    require(len(candidate.layers) == len(reference.layers), 'SCHEMA: prefix layer count mismatch')
    rows = []
    for index, (a, b) in enumerate(zip(candidate.layers, reference.layers)):
        item = {'layer': index}
        for axis, x, y in (('K', a.keys, b.keys), ('V', a.values, b.values)):
            require(x.shape == y.shape and x.dtype == y.dtype, 'SCHEMA: prefix KV identity geometry mismatch')
            finite(x, 'prefix candidate ' + axis); finite(y, 'prefix reference ' + axis)
            diff = x.float() - y.float()
            item[axis] = {'max_abs': float(diff.abs().max()),
                          'relative_L2': float(diff.norm() / y.float().norm().clamp_min(1e-30)),
                          'bitwise_equal': bool(torch.equal(x.contiguous().view(torch.uint8),
                                                            y.contiguous().view(torch.uint8)))}
        rows.append(item)
    return {'layers': rows, 'all_layers_KV_bitwise_equal': all(v[a]['bitwise_equal'] for v in rows for a in ('K', 'V')),
            'max_abs_K': max(v['K']['max_abs'] for v in rows),
            'max_abs_V': max(v['V']['max_abs'] for v in rows),
            'numerical_difference_alone_is_not_failure': True}


def generation(model, processor, batch, cache, hybrid, native, cap, forward_logits):
    import torch
    from attention_bridge import attention_context
    from m1_protocol import m1_generate_native, m1_generate_cached
    events, first_logits = [], []
    def before(module, positional, kwargs):
        ids = kwargs.get('input_ids')
        if ids is None and positional and isinstance(positional[0], torch.Tensor):
            ids = positional[0]
        past = kwargs.get('past_key_values')
        positions = kwargs.get('position_ids'); cp = kwargs.get('cache_position')
        event = {'input_tokens': int(ids.shape[-1]) if ids is not None else None,
                 'input_ids_sha256': tensor_sha(ids) if ids is not None else None,
                 'cache_tokens_before': int(past.get_seq_length()) if past is not None else 0,
                 'cache_type': type(past).__name__ if past is not None else None,
                 'position_ids': tensor_contract(positions) if positions is not None else None,
                 'cache_position': tensor_contract(cp) if cp is not None else None}
        events.append(event)
    def after(module, positional, output):
        finite(output.logits, 'generation actual outer logits')
        require(events, 'SCHEMA: output observed without a forward event')
        events[-1]['outer_logits_finite'] = True
        events[-1]['last_raw_top1'] = int(output.logits[0, -1].argmax())
        if not first_logits:
            first_logits.append(output.logits[0, -1].detach().float().cpu())
    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after)
    try:
        context = attention_context(model, cache=cache) if hybrid else nullcontext()
        with context:
            sequence = (m1_generate_native(model, batch, max_new_tokens=cap) if native else
                        m1_generate_cached(model, batch, cache, max_new_tokens=cap))
    finally:
        post.remove(); pre.remove()
    require(sequence.ndim == 2 and sequence.shape[0] == 1 and sequence.shape[-1] >= batch.full_length,
            'SCHEMA: native generation sequence shape differs')
    require(torch.equal(sequence[:, :batch.full_length], batch.ids), 'IDENTITY: generated history prefix changed')
    ids = sequence[0, batch.full_length:].detach().cpu().tolist()
    require(ids and len(ids) <= cap and events and first_logits, 'SCHEMA: missing/oversized generated answer')
    expected_first = batch.ids if native else batch.ids[:, batch.boundary:]
    first = events[0]
    expected_positions = batch.positions if native else batch.positions[:, batch.boundary:]
    expected_cache_length = 0 if native else batch.boundary
    provided_positions = first['position_ids']
    expected_cache_positions = torch.arange(expected_cache_length, batch.full_length,
                                            device=batch.ids.device, dtype=torch.int64)
    provided_cache_positions = first['cache_position']
    cache_position_match = (provided_cache_positions is None or
                            provided_cache_positions == tensor_contract(expected_cache_positions))
    position_match = (provided_positions is None or
                      provided_positions == tensor_contract(expected_positions))
    first_contract = {'expected_input_tokens': int(expected_first.shape[-1]),
                      'expected_input_ids_sha256': tensor_sha(expected_first),
                      'actual_input_tokens': first['input_tokens'],
                      'actual_input_ids_sha256': first['input_ids_sha256'],
                      'actual_first_call_matches_expected_suffix_or_full': first['input_ids_sha256'] == tensor_sha(expected_first),
                      'expected_cache_tokens_before': expected_cache_length,
                      'actual_cache_tokens_before': first['cache_tokens_before'],
                      'expected_explicit_position_ids': tensor_contract(expected_positions),
                      'actual_explicit_position_ids': provided_positions,
                      'provided_position_ids_match_original_positions': position_match,
                      'expected_explicit_cache_position': tensor_contract(expected_cache_positions),
                      'actual_explicit_cache_position': provided_cache_positions,
                      'provided_cache_position_matches_original_append_range': cache_position_match,
                      'absent_cache_position_use_original_native_forward_cache_position_inference': provided_cache_positions is None,
                      'absent_position_ids_use_original_native_forward_position_inference': provided_positions is None,
                      'first_token_derived_from_this_branch': True,
                      'native_first_token_or_logits_injected': False}
    # These are input/position identities, not numerical-logit tolerances.
    require(first_contract['actual_first_call_matches_expected_suffix_or_full']
            and first['input_tokens'] == int(expected_first.shape[-1])
            and first['cache_tokens_before'] == expected_cache_length and position_match and cache_position_match,
            'IDENTITY: actual first generation call differs from declared full/suffix contract: '
            + json.dumps(first_contract, sort_keys=True))
    return {'output_token_ids': ids, 'output_tokens': len(ids),
            'raw_decode_skip_special_tokens': processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            'raw_decode_with_special_tokens': processor.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'actual_first_generation_call': first_contract,
            'generation_first_raw_logits_vs_separate_forward': logits_difference(first_logits[0], forward_logits),
            'generation_forward_events': events,
            'all_actual_generation_outer_logits_finite': all(x.get('outer_logits_finite') for x in events)}


def run_arm(name, model, processor, batch, spec, cap):
    import torch
    from attention_bridge import attention_context
    from cache_bridge import HybridVisualCache
    from m1_protocol import (m1_native_full_forward, m1_prefix_forward, m1_forward_suffix,
                             clone_m1_prefix_cache)
    prototype = output = measure_cache = generate_cache = None
    branch = {'name': name, 'status': 'RUNNING', 'compression_used': False,
              'measurement_generation_share_mutated_cache': False,
              'diagnostic_not_storage_or_speed_benchmark': True}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated(); start = time.perf_counter()
    try:
        if name in (ARMS[0], ARMS[2]):
            output = m1_native_full_forward(model, batch, logits_to_keep=1)
            finite(output.logits, name + '/native complete-prompt logits')
            if name == ARMS[0]:
                last_logits = output.logits[0, -1].detach().float().cpu()
            source = output.past_key_values
            require(int(source.get_seq_length()) == batch.full_length, 'SCHEMA: full native cache length differs')
            source.crop(batch.boundary)
            prototype = clone_m1_prefix_cache(source)
            branch['prefix_origin'] = 'complete_prompt_native_prefill_then_actual_crop_to_boundary'
            branch['old_custom_R7_runner_equivalence_claimed'] = False
            output = source = None
        else:
            output = m1_prefix_forward(model, batch)
            finite(output.logits, name + '/native prefix logits')
            prototype = clone_m1_prefix_cache(output.past_key_values)
            output = None
            branch['prefix_origin'] = 'native_image_prefix_only_no_question_tokens'
        validate_prefix(prototype, batch.boundary, spec, name)
        if name != ARMS[0]:
            measure_cache = clone_m1_prefix_cache(prototype)
            if name == ARMS[3]:
                measure_cache = HybridVisualCache(measure_cache, batch.visual_positions)
            context = attention_context(model, cache=measure_cache) if name == ARMS[3] else nullcontext()
            with context:
                output = m1_forward_suffix(model, batch, measure_cache, logits_to_keep=1)
            finite(output.logits, name + '/complete suffix last logits')
            last_logits = output.logits[0, -1].detach().float().cpu()
            require(int(measure_cache.get_seq_length()) == batch.full_length, 'SCHEMA: suffix did not append complete question')
            output = measure_cache = None
        # Rebuild from the untouched prototype; never recycle the suffix cache.
        if name != ARMS[0]:
            generate_cache = clone_m1_prefix_cache(prototype)
            if name == ARMS[3]:
                generate_cache = HybridVisualCache(generate_cache, batch.visual_positions)
        branch.update(generation(model, processor, batch, generate_cache,
                                 name == ARMS[3], name == ARMS[0], cap, last_logits))
        branch.update(status='PASS', last_forward_raw_top1=int(last_logits.argmax()),
                      last_forward_logits_finite=True, prefix_all_layers_KV_finite=True,
                      FP16_hybrid_reader='merge_original_order_then_original_HF_SDPA' if name == ARMS[3] else None)
        return branch, prototype, last_logits
    except Exception as exc:
        branch.update(status='FAILED_RUN', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        return branch, None, None
    finally:
        output = measure_cache = generate_cache = None
        torch.cuda.synchronize()
        branch['diagnostic_wall_ms'] = (time.perf_counter() - start) * 1000
        branch['CUDA_memory'] = {'allocated_before_bytes': allocated,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                'native_prototypes_retained_not_deployment_peak': True}
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    entries = validate_config(config)
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), *[ROOT/'code'/name for name in
        ('m1_protocol.py', 'model_registry.py', 'dataset_registry.py', 'evaluate.py',
         'cache_bridge.py', 'attention_bridge.py', 'ownership.py')]]
    source_files = {str(path): sha(path) for path in source_paths}
    pinned = config.get('baseline_frozen_sources', {})
    expected_pins = {'code/m1_protocol.py', 'code/cache_bridge.py', 'code/attention_bridge.py', 'code/ownership.py'}
    normalized_pins = {key.replace('\\', '/'): value for key, value in pinned.items()}
    require(set(normalized_pins) == expected_pins, 'IDENTITY: exact four baseline source pins required')
    require(all(sha(ROOT/key) == digest for key, digest in normalized_pins.items()),
            'IDENTITY: baseline frozen source SHA differs')
    manifest = {'schema': SCHEMA + '.manifest', 'config': config, 'config_sha256': sha(args.config),
                'driver_sha256': sha(__file__), 'source_sha256': source_files,
                'started_unix': time.time(), 'expected_QA': 8, 'expected_arms_per_QA': 4,
                'references_read': False, 'quality_scored': False, 'algorithm_or_codec_tested': False,
                'native_prototypes_retained_for_protocol_diagnostics': True,
                'numerical_tolerance_gate': None,
                'scope': 'M1 FP16 native vs prefix/crop/Hybrid protocol controls; no benchmark500 or R7 codec',
                'old_custom_R7_runner_equivalence_claimed': False}
    manifest_written = False; rows_written = 0; counts = {}; load_error = None
    records_path = args.output/'records.jsonl'
    try:
        import torch
        import transformers
        import tokenizers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from dataset_registry import load_inputs
        from evaluate import model_identity
        from model_registry import get_model_spec
        from attention_bridge import language_attention_modules
        from m1_protocol import (prepare_m1_inputs, m1_native_full_forward, clone_m1_prefix_cache)
        torch.set_grad_enabled(False); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available(), 'HARDWARE: actual CUDA required')
        torch.cuda.set_device(0)
        require(torch.cuda.get_device_capability() == (8, 6), 'HARDWARE: RTX3090 SM86 required')
        spec = get_model_spec('M1')
        require(config.get('model_revision', spec['revision']) == spec['revision'], 'IDENTITY: locked M1 revision required')
        identity = model_identity(config, spec)
        inputs, input_files = [], []
        for entry in entries:
            digest = sha(entry['path'])
            require(entry.get('sha256', digest) == digest, 'IDENTITY: configured input SHA differs')
            selected = load_inputs(entry['path'])[:4]
            require(len(selected) == 4 and all(row['dataset'] == entry['dataset'] for row in selected),
                    'IDENTITY: input dataset/count differs')
            inputs.extend(selected)
            input_files.append({**entry, 'sha256': digest, 'selected_ids': [x['id'] for x in selected]})
        require(len(inputs) == len({x['id'] for x in inputs}) == 8, 'IDENTITY: expected8 unique QA')
        processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(config['model_path'], dtype=torch.float16,
                    device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        require(all(parameter.dtype == torch.float16 for parameter in model.parameters())
                and not getattr(model, 'is_quantized', False), 'IDENTITY: original FP16 model weights required')
        require(int(model.generation_config.num_beams or 1) == 1, 'SCHEMA: native beam-one greedy required')
        _, modules = language_attention_modules(model)
        require(len(modules) == spec['layers'], 'IDENTITY: M1 language layers differ')
        for obj in (type(model), type(processor), type(processor.tokenizer)):
            path = Path(inspect.getfile(obj)); source_files[str(path)] = sha(path)
        manifest.update(model_identity=identity, model_spec=spec, input_files=input_files,
                        expected_ids=[x['id'] for x in inputs], torch_version=torch.__version__,
                        transformers_version=transformers.__version__, python_version=platform.python_version(),
                        tokenizer_class=type(processor.tokenizer).__module__+'.'+type(processor.tokenizer).__name__,
                        tokenizer_is_fast=bool(getattr(processor.tokenizer,'is_fast',False)),
                        tokenizers_version=tokenizers.__version__,
                        decode_overrides={'clean_up_tokenization_spaces':False,
                                          'both_skip_special_tokens_true_and_false_saved':True},
                        CUDA_version=torch.version.cuda, hardware=torch.cuda.get_device_name(),
                        tf32=False, seed=0, weight_dtype='float16', attention_implementation='sdpa',
                        processor_file_hashes={x['file']: x['sha256'] for x in identity['files']
                                               if not x['file'].endswith('.safetensors')},
                        generation_native=model.generation_config.to_dict(),
                        generation_overrides={'do_sample': False, 'use_cache': True,
                                              'max_new_tokens': config['max_new_tokens']},
                        source_sha256=source_files)
        write_new(args.output/'manifest.json', manifest); manifest_written = True
        with records_path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in inputs:
                record = {'schema': SCHEMA + '.record', 'id': row['id'], 'dataset': row['dataset'],
                          'source_cluster_id': row['source_cluster_id'], 'status': 'RUNNING',
                          'original_image_sha256': row['original_image_sha256'], 'arms': [],
                          'references_read': False, 'quality_scored': False}
                batch = alternate = alt_output = alt_prefix = None; prototypes = {}; logits = {}
                try:
                    require(sha(row['image_path']) == row['original_image_sha256'], 'IDENTITY: original image SHA differs')
                    with Image.open(row['image_path']) as original:
                        record['original_dimensions'] = list(original.size); image = original.convert('RGB')
                    rendered = row['question'] + SUFFIX
                    batch = prepare_m1_inputs(processor, image, rendered, image_token_id=spec['image_token_id'], max_patches=8)
                    alternate_question = ('Describe the objects.' if row['question'] == 'What is visible in the image?'
                                          else 'What is visible in the image?')
                    alternate = prepare_m1_inputs(processor, image, alternate_question+SUFFIX,
                                                 image_token_id=spec['image_token_id'], max_patches=8)
                    require(batch.suffix_length > 0 and batch.boundary == int(batch.visual_positions[-1])+1,
                            'SCHEMA: invalid native visual boundary/suffix')
                    require(alternate.boundary == batch.boundary
                            and torch.equal(batch.ids[:, :batch.boundary], alternate.ids[:, :alternate.boundary])
                            and torch.equal(batch.visual_positions, alternate.visual_positions)
                            and torch.equal(batch.positions[:, :batch.boundary], alternate.positions[:, :alternate.boundary]),
                            'IDENTITY: future query changes image-prefix token/position identity')
                    visual_fields = {}
                    for key in set(batch.inputs) | set(alternate.inputs):
                        if key in ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids'):
                            continue
                        require(key in batch.inputs and key in alternate.inputs, 'IDENTITY: future query changes multimodal input keys')
                        a, b = batch.inputs[key], alternate.inputs[key]
                        if isinstance(a, torch.Tensor):
                            require(torch.equal(a, b), 'IDENTITY: future query changes visual processor tensor ' + key)
                            visual_fields[key] = tensor_contract(a)
                        else:
                            require(a == b, 'IDENTITY: future query changes visual processor field ' + key)
                    template = processor.chat_template
                    record.update(raw_question=row['question'], rendered_question=rendered,
                        question_suffix=SUFFIX, full_prompt=batch.prompt,
                        full_prompt_hash=hashlib.sha256(batch.prompt.encode('utf-8')).hexdigest(),
                        chat_template_hash=hashlib.sha256(template.encode('utf-8')).hexdigest() if isinstance(template,str) else json_sha(template),
                        input_ids=tensor_contract(batch.ids, True), question_suffix_ids=tensor_contract(batch.ids[:, batch.boundary:], True),
                        attention_mask=tensor_contract(batch.mask, True), visual_positions=tensor_contract(batch.visual_positions, True),
                        native_position_ids=tensor_contract(batch.positions, True), compression_boundary=batch.boundary,
                        actual_visual_token_count=int(batch.visual_positions.numel()), visual_processor_inputs=visual_fields,
                        future_query_prefix_identity={'replacement_question': alternate_question,
                            'prefix_input_ids_and_visual_positions_and_position_ids_exact': True,
                            'visual_processor_fields_exact': True, 'replacement_suffix_ids': tensor_contract(alternate.ids[:, alternate.boundary:])},
                        common_input_contract_sha256=json_sha({'input_ids':tensor_sha(batch.ids),'image':row['original_image_sha256'],
                            'visual_positions':tensor_sha(batch.visual_positions),'position_ids':tensor_sha(batch.positions),
                            'boundary':batch.boundary,'processor_files':manifest['processor_file_hashes']}))
                    for name in ARMS:
                        branch, prototype, last = run_arm(name, model, processor, batch, spec, config['max_new_tokens'])
                        record['arms'].append(branch)
                        if prototype is not None: prototypes[name] = prototype
                        if last is not None: logits[name] = last
                        print(json.dumps({'id':row['id'],'arm':name,'status':branch['status']}), flush=True)
                    record['prefix_KV_comparisons'] = {}
                    record['last_question_logits_comparisons'] = {}
                    for ai in range(len(ARMS)):
                        for bi in range(ai+1, len(ARMS)):
                            left, right = ARMS[ai], ARMS[bi]; key = right+'_vs_'+left
                            if left in prototypes and right in prototypes:
                                record['prefix_KV_comparisons'][key] = cache_difference(prototypes[right], prototypes[left])
                            if left in logits and right in logits:
                                record['last_question_logits_comparisons'][key] = logits_difference(logits[right], logits[left])
                    # Replace future query in an actual full native prefill and
                    # compare only its cropped prefix. No answer is generated.
                    alt_output = m1_native_full_forward(model, alternate, logits_to_keep=1)
                    finite(alt_output.logits, 'alternate future-query native full logits')
                    alt_output.past_key_values.crop(batch.boundary)
                    alt_prefix = clone_m1_prefix_cache(alt_output.past_key_values)
                    validate_prefix(alt_prefix, batch.boundary, spec, 'alternate full prompt cropped prefix')
                    if ARMS[0] in prototypes:
                        record['future_query_prefix_identity']['actual_full_prefill_cropped_prefix_KV'] = cache_difference(alt_prefix, prototypes[ARMS[0]])
                    record['generation_sequence_comparisons'] = {}
                    by_arm = {x['name']:x for x in record['arms']}
                    for name in ARMS[1:]:
                        if by_arm[name]['status'] == by_arm[ARMS[0]]['status'] == 'PASS':
                            x,y=by_arm[name]['output_token_ids'],by_arm[ARMS[0]]['output_token_ids']
                            at=next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)) if len(x)!=len(y) else None)
                            record['generation_sequence_comparisons'][name+'_vs_A']={'tokens_exact_equal':x==y,'first_difference_index':at}
                    record['status'] = 'PASS' if all(x['status']=='PASS' for x in record['arms']) else 'COMPLETE_WITH_FAILURES'
                    record['critical_input_contracts_shared_by_all_four_paths'] = {
                        'same_original_image_processor_and_template': True, 'same_full_prompt_ids_positions_boundary': True,
                        'same_suffix_ids_original_positions_and_generation_cap': True}
                except Exception as exc:
                    record.update(status='FAILED_RUN', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                finally:
                    batch=alternate=alt_output=alt_prefix=None; prototypes.clear(); logits.clear()
                    gc.collect(); torch.cuda.empty_cache()
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                rows_written += 1; counts[record['status']] = counts.get(record['status'],0)+1
        require(all(sha(path)==digest for path,digest in source_files.items()), 'IDENTITY: diagnostic source changed during run')
        require(sha(args.config)==manifest['config_sha256'], 'IDENTITY: config changed during run')
        require(all(sha(entry['path']) == entry['sha256'] for entry in input_files),
                'IDENTITY: input registry file changed during run')
    except Exception as exc:
        load_error = {'error_type':type(exc).__name__, 'error':str(exc), 'traceback':traceback.format_exc()}
    finally:
        if not manifest_written:
            manifest['initialization_error'] = load_error
            write_new(args.output/'manifest.json', manifest)
        if not records_path.exists():
            records_path.open('x',encoding='utf-8').close()
        complete = {'schema':SCHEMA+'.complete', 'status':'PASS_PROTOCOL_DIAGNOSTIC_COMPLETE'
                    if load_error is None and rows_written==8 and counts=={'PASS':8} else 'DIAGNOSTIC_COMPLETE_WITH_FAILURES',
                    'expected_QA':8, 'recorded_QA':rows_written, 'counts':counts, 'arms_per_QA':4,
                    'load_or_final_identity_error':load_error, 'references_read':False,'quality_scored':False,
                    'numerical_differences_alone_not_failure':True, 'old_R7_custom_runner_equivalence_claimed':False,
                    'raw_files':{name:sha(args.output/name) for name in ('manifest.json','records.jsonl')},
                    'completed_unix':time.time()}
        write_new(args.output/'complete.json',complete)
        print(json.dumps({'status':complete['status'],'recorded_QA':rows_written,'counts':counts,'error':load_error}),flush=True)
    return 0 if complete['status']=='PASS_PROTOCOL_DIAGNOSTIC_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
