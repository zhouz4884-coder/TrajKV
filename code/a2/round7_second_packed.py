"""Round7 LLaVA-Next arbitrary-real-N first-question packed E2E runner.

Uses the separately smoke-verified second-session HF bridge, actual PQ K2 /
whole-image tail-packed binary V1 and compact two-stage residuals. Real Nv is never padded.
The static 32x32 allocation and identity factors are independent of InternVL.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_template(processor, config):
    """Use the same pinned official-template injection as the native/Full smoke."""
    installed = getattr(processor, 'chat_template', None) or getattr(processor.tokenizer, 'chat_template', None)
    path = config.get('official_chat_template_path')
    official = None
    if path:
        if sha(path) != config['official_chat_template_sha256']:
            raise ValueError('Pinned official processor template bytes changed')
        official = json.loads(Path(path).read_text(encoding='utf-8-sig'))['chat_template']
    injected = not bool(installed)
    if injected:
        if not official:
            raise ValueError('LLaVA-Next requires an installed or pinned official chat template')
        processor.chat_template = official
    actual = getattr(processor, 'chat_template', None) or getattr(processor.tokenizer, 'chat_template', None)
    return dict(official_template_injected_if_missing=injected,
        actual_template_sha256=hashlib.sha256(str(actual).encode()).hexdigest(),
        template_file_sha256=sha(path) if path else None,
        actual_template_matches_pinned=actual == official if official is not None else None,
        template_source='pinned official artifact' if injected else 'locally installed processor/tokenizer template',
        template_provenance=config.get('official_chat_template_source'))


def verify_identity_and_rank(variant, geometry):
    """Reject transferred learned metadata; verify the source-independent ranks."""
    factor = json.loads(Path(variant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    expected = {'layers':32, 'kv_heads':32, 'head_dim':128}
    if factor.get('geometry') != expected or factor.get('calibration_mode') != 'identity_no_task_data':
        raise ValueError('Second-model factors must be declared independent 32x32 identities')
    if factor.get('cases', 0) != 0 or factor.get('images', 0) != 0 or factor.get('labels_used', False):
        raise ValueError('Identity metadata declaration contains task calibration or labels')
    if factor.get('key_gain_mode') != 'none' or factor.get('value_group_size') != 'all' or factor.get('parameter_count') != 3072:
        raise ValueError('Second-model identity metadata settings differ from the declared format')
    for key, value in (('key_head_factors', 1.), ('value_head_factors', 1.), ('value_center_offsets', 0.)):
        array = factor.get(key)
        if not isinstance(array, list) or len(array) != 32 or any(len(row) != 32 or any(float(x) != value for x in row) for row in array):
            raise ValueError('Second-model metadata numerical values are not 32x32 identities: ' + key)
    if factor.get('encoder_book_sha256') != sha(variant['direction_book_path']):
        raise ValueError('Second-model identity factor artifact targets a different book')
    ranking = variant.get('selected_units')
    ranking_kind = 'explicit stages; no unit ranking'
    if ranking is not None:
        spread = [int(f'{i:010b}'[::-1], 2) for i in range(1024)]
        shuffled = list(range(1024))
        random.Random(1212).shuffle(shuffled)
        candidates = {name: [[axis, index // 32, index % 32] for index in order for axis in ('k', 'v')]
                      for name, order in (('spread', spread), ('random1212', shuffled))}
        candidates.update({name: [[axis, index // 32, index % 32] for axis in axes for index in spread]
                           for name, axes in (('axisfirst_Vspread', ('v', 'k')),
                                              ('axisfirst_Kspread', ('k', 'v')))})
        matched = [name for name, order in candidates.items() if ranking == order]
        if not matched:
            raise ValueError('Second-model ranking must exactly match spread/random1212/axisfirst_Vspread/axisfirst_Kspread')
        ranking_kind = matched[0]
    return dict(identity_numerical_values_verified=True, factor_sha256=sha(variant['metadata_factor_path']),
        factor_geometry=expected, factor_calibration_cases=0, model_config_bound_separately=True,
        source_independent_ranking=ranking_kind,
        ranking_sha256=hashlib.sha256(json.dumps(ranking, separators=(',', ':')).encode()).hexdigest() if ranking is not None else None)


def run_packed(model, processor, features, visual, args, variant, full_trace):
    import torch
    from round7_tail_key_bank import TailKeyBank
    from round6_second_session import (bank_layer_proxy, bank_attention, forward,
        consume, stop_ids)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); started = time.perf_counter()
    inputs = {key:value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
              for key,value in features.items()}
    visual = visual.to(model.device)
    out = forward(model, **inputs)
    cache = out.past_key_values
    boundary = int(visual[-1]) + 1
    suffix = inputs['input_ids'][:,boundary:].clone()
    if not suffix.numel():
        raise ValueError('Compression must precede a complete nonempty question suffix')
    bank = None
    try:
        TailKeyBank.settings = variant
        bank = TailKeyBank(bank_layer_proxy(model), cache, visual, 'round7_tail_key',
            Path(args.legacy_scripts), Path(args.kivi_source), keep_reference=False)
        nv, nl, nh = int(visual.numel()), len(bank.states), bank.states[0].text_k.shape[1]
        words_per_channel = (nv + 31) // 32
        physical_v_slots = words_per_channel * 32
        expected_v_code_bytes = nl * nh * 128 * words_per_channel * 4
        actual_v_code_bytes = sum(state.v_code.numel() * state.v_code.element_size() for state in bank.states)
        if actual_v_code_bytes != expected_v_code_bytes:
            raise RuntimeError('Actual V bit-word owner differs from ceil(real Nv/32) words per channel')
        if any(hasattr(state, name) for state in bank.states for name in ('visual_k', 'visual_v')):
            raise RuntimeError('Tail packed deployment retained FP16 visual source tensors')
        if any(state.visual_pos.numel() != nv or not torch.equal(state.visual_pos, visual) for state in bank.states):
            raise RuntimeError('Physical V tail bits changed the real visual token positions')
        for state in bank.states:
            state.text_length = int((state.text_pos[:state.text_length] < boundary).sum())
            state.length = boundary
        bank.turn, bank.phase, bank.pass_kind = 1, 'question_prefill', 'committed'
        bank.refine_layers = set(range(len(bank.states)))
        with bank_attention(model):
            out, logical = consume(model, cache, suffix, boundary)
        logits = out.logits[:,-1].clone()
        initial_owner = bank.owned()
        bank.on_turn_begin(1, logical)
        del out, inputs
        ids, comparisons, prefix_matches = [], [], True
        stops = stop_ids(model, processor)
        first_logits = logits.detach().float().cpu()
        for step in range(args.max_tokens):
            token = int(logits.argmax(-1))
            if full_trace is not None and prefix_matches and step < len(full_trace['logits']):
                reference = full_trace['logits'][step]
                error = logits.detach().float().cpu() - reference
                comparisons.append(dict(step=step, max_abs_error=float(error.abs().max()),
                    relative_l2=float(error.norm()/reference.norm().clamp_min(1e-20)),
                    argmax_same=token == full_trace['token_ids'][step]))
                prefix_matches = token == full_trace['token_ids'][step]
            ids.append(token)
            if token in stops or step + 1 == args.max_tokens:
                break
            bank.phase = 'decode'
            with bank_attention(model):
                out, logical = consume(model, cache, torch.tensor([[token]], device=model.device), logical)
            logits = out.logits[:,-1].clone()
            del out
        bank.on_turn_end(1, logical)
        torch.cuda.synchronize()
        encoder = int(bank.encoder_parameter_bytes)
        complete = sum(int(initial_owner.get(key,0)) for key in
            ('visual_base','visual_residual','selector_metadata','visual_position')) + encoder
        actual_bits = 16 * complete / bank.visual_full_bytes
        if actual_bits > float(variant.get('budget_bits',1.8)) + 1e-12:
            raise RuntimeError('Second-backbone complete packed owner exceeds its budget')
        if initial_owner.get('full_visual_retained') or initial_owner.get('diagnostic_fp16_reference'):
            raise RuntimeError('Packed deployment retained source visual K/V')
        result = dict(variant=variant['name'], variant_config=variant, method='round7_tail_key',
            protocol='first_question_image_boundary_free_running',
            turns=[dict(turn=1,token_ids=ids,answer=processor.tokenizer.decode(ids,skip_special_tokens=True).strip())],
            visual_tokens=int(visual.numel()),padded_visual_tokens=0,owned_initial=initial_owner,
            owned_final=bank.owned(), visual_positions_exact_original=True,
            value_words_per_channel=words_per_channel, value_physical_bit_slots_per_channel=physical_v_slots,
            value_unused_tail_bits_per_channel=physical_v_slots-nv,
            value_code_actual_bytes=actual_v_code_bytes,
            value_tail_padding_actual_bytes=16*(physical_v_slots-nv)*nl*nh,
            value_tail_padding_scope='unused physical bits in the final channel word only; no extra tokens, positions or centroid-fit samples',
            encoder_calibration_parameters_bytes=encoder,actual_visual_bits_including_encoder=actual_bits,
            complete_visual_owner_bytes=complete,peak_allocated=torch.cuda.max_memory_allocated(),
            session_ms=1000*(time.perf_counter()-started),answers_visible_to_runner=False,
            initial_logits_sha256=hashlib.sha256(first_logits.numpy().tobytes()).hexdigest(),
            compression_point='after final visual token, before complete first-question suffix',
            first_question_suffix_replay_tokens=int(suffix.shape[-1]),initial_logits_shared_fp16=False,
            question_state_scope='recomputed_under_codec',task_accuracy_eligible=True,
            full_same_prefix_logit_comparisons=comparisons,
            full_complete_tokens_equal=ids==full_trace['token_ids'] if full_trace is not None else None,
            full_comparison_scope='free-running outputs; same-prefix logits only until first emitted-token divergence',
            calibration_scope='new 32x32 identity factors and source-independent static ranking; no second-model task calibration',
            lowbit_deployment_candidate=True)
        result.update(bank.result_metadata())
        result['read_events'] = bank.event_rows()
        result['read_ledger_exact'] = sum(e.get('selection_bytes',0)+e.get('residual_bytes',0)
            for e in result['read_events']) == bank.extra_read_bytes
        if not result['read_ledger_exact']:
            raise RuntimeError('Packed read ledger disagrees with actual calls')
        return result
    finally:
        if bank is not None:
            bank.close()
        del bank, cache
        gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from round6_second_session import first_question_features, model_geometry, run_first_question
    a = SimpleNamespace(**config['arguments'])
    if int(getattr(a,'turns',1)) != 1:
        raise ValueError('Only independently serialized first questions are supported')
    a.max_tokens = int(getattr(a,'max_tokens',32))
    if a.max_tokens <= 0:
        raise ValueError('A positive natural free-generation token limit is required')
    sys.path.insert(0,str(Path(a.legacy_scripts)))
    torch.set_num_threads(4);torch.manual_seed(1212);torch.backends.cuda.matmul.allow_tf32=False
    rows = json.loads(Path(a.samples).read_text())
    rows = rows[int(config.get('start',0)):int(config.get('start',0))+int(config['count'])]
    if len(rows)!=int(config['count']) or len({r['id'] for r in rows})!=len(rows) or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Missing/duplicate second-model cases')
    variants = config['variants']
    full_indices = [i for i,v in enumerate(variants) if v['method']=='full']
    if len(full_indices)!=1 or any(v['method'] not in ('native_hf','full','round7_tail_key') for v in variants):
        raise ValueError('Require one Full reference and supported packed codecs only')
    first_compressed = min((i for i,v in enumerate(variants) if v['method']=='round7_tail_key'),default=len(variants))
    if full_indices[0] >= first_compressed:
        raise ValueError('Full reference must precede packed free generation')
    native_indices = [i for i,v in enumerate(variants) if v['method']=='native_hf']
    if len(native_indices) > 1 or native_indices and native_indices[0] > full_indices[0]:
        raise ValueError('An optional single native HF run must precede Full')
    if len({v['name'] for v in variants}) != len(variants):
        raise ValueError('Variant names must be unique')
    model_path = Path(a.model) / 'config.json'
    model_config_hash = sha(model_path)
    model_config = json.loads(model_path.read_text(encoding='utf-8-sig'))
    if model_config.get('quantization_config') or model_config.get('model_type') != 'llava_next':
        raise ValueError('Actual FP16 LLaVA-Next weights are required for this second-backbone runner')
    if not config.get('expected_model_config_sha256') or model_config_hash != config['expected_model_config_sha256']:
        raise ValueError('Installed second-backbone model identity differs from the declared config SHA256')
    processor = AutoProcessor.from_pretrained(a.model,local_files_only=True)
    template = install_template(processor, config)
    model = AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    geometry = model_geometry(model)
    if (geometry['layers'],geometry['kv_heads'],geometry['head_dim']) != (32,32,128):
        raise ValueError('This static second-backbone configuration is specifically L32/Hkv32/D128')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.dtype != torch.float16:
        raise ValueError('The model must actually use FP16 weights')
    external = {v[key]:sha(v[key]) for v in variants for key in ('direction_book_path','metadata_factor_path','key_metric_path') if v.get(key)}
    if config.get('official_chat_template_path'):
        external[config['official_chat_template_path']] = sha(config['official_chat_template_path'])
    identity_validation = {}
    for variant in variants:
        if variant['method']!='round7_tail_key':
            continue
        identity_validation[variant['name']] = verify_identity_and_rank(variant, geometry)
        if variant.get('v_group_size') != 'all':
            raise ValueError('This arbitrary-real-N second-model comparison uses whole-image V metadata')
    image_inputs = []
    for row in rows:
        if any(key in row for key in ('answer', 'answers', 'references', 'ground_truth')):
            raise ValueError('Inference samples must not contain reference answers')
        paths = row.get('remote_images', row.get('images', []))
        hashes = [sha(path) for path in paths]
        if row.get('hashes') and hashes != row['hashes']:
            raise ValueError('Actual source image bytes differ from the sample manifest')
        image_inputs.append(dict(id=row['id'], image_id=row['image_id'], image_sha256=hashes))
    manifest = dict(schema='trajkv.round7.second-backbone-tail-packed.v1',config=config,config_sha256=sha(args.config),
        samples_sha256=sha(a.samples),selected_ids=[r['id'] for r in rows],geometry=geometry,
        model_config_sha256=model_config_hash, model_identity_scope='actual installed model config SHA256; source-independent factor/rank provenance listed separately',
        external_codec_inputs=external, identity_factor_rank_validation=identity_validation,
        prompt_template=template, image_inputs=image_inputs,
        answers_visible_to_runner=False,model_weights_frozen=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        sources={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),Path(__file__).with_name('round6_second_session.py'),
            *[Path(__file__).with_name(name) for name in ('round7_tail_key_bank.py','round7_tail_key_attention.py','round7_tail_key_codec.py',
                'round7_key_codec.py','round7_key_layout.py','round7_key_artifacts.py','round6_depth_codec.py','fast_wht.py','quantizers.py',
                'affine_rate_bank.py','masked_bank.py','metadata_direction_bank.py','learned_direction_bank.py')]]},
        weight_quantization=False, weight_dtype=str(model.dtype), text_KV='FP16', visual_tokens_padded=False,
        scope='unfitted second-backbone development transfer; actual packed E2E; no blind-confirmation claim')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records, paired_checks, visual_counts = 0, [], []
    with torch.inference_mode(), (args.output/'records.jsonl').open('x') as stream:
        for row in rows:
            features,visual,prompt = first_question_features(processor,model,row)
            if visual.numel() <= 0 or prompt['padded_visual_tokens'] != 0:
                raise ValueError('A nonempty original real visual-token segment is required')
            visual_counts.append(int(visual.numel()))
            native,full,groups = None,None,{}
            for variant in variants:
                if variant['method']=='round7_tail_key':
                    result = run_packed(model,processor,features,visual,a,variant,full)
                    group = variant.get('matched_owner_group')
                    current = (result['complete_visual_owner_bytes'],result['depth_residual_stages'],
                        result['depth_residual_K_stages'],result['depth_residual_V_stages'])
                    if group is not None and group in groups and groups[group]!=current:
                        raise RuntimeError('Width/pair controls differ in actual complete bytes/stage axis costs')
                    if group is not None:
                        groups[group] = current
                else:
                    result,trace = run_first_question(model,processor,features,visual,a,variant,native)
                    if variant['method']=='native_hf':
                        native=trace
                    elif variant['method']=='full':
                        full=trace
                        if native is not None and not result['native_complete_tokens_equal']:
                            raise RuntimeError('Full free tokens diverged from native identity smoke')
                result.update(session_id=row['id'],image_id=row['image_id'],dataset=row.get('dataset','docvqa'),
                    source_document_id=row.get('source_document_id'),source_original_image_id=row.get('source_original_image_id'),
                    split=row.get('split'),model=a.model,prompt_metadata=prompt,
                    model_config_sha256=model_config_hash, visual_tokens_mod32=int(visual.numel()) % 32)
                stream.write(json.dumps(result)+'\n');stream.flush();records+=1
                print(json.dumps({'done':records,'id':row['id'],'variant':variant['name'],'visual_tokens':int(visual.numel()),
                    'actual_bits':result['actual_visual_bits_including_encoder'],'answer':result['turns'][0]['answer']}),flush=True)
            paired_checks.append({'id':row['id'],'explicit_matched_owner_groups_verified':True,'groups':groups})
    for path,digest in external.items():
        if sha(path)!=digest:
            raise RuntimeError('Frozen second-backbone encoder input changed')
    if sha(model_path) != model_config_hash or sha(a.samples) != manifest['samples_sha256']:
        raise RuntimeError('Second-backbone model config or sample manifest changed during the run')
    complete=dict(status='COMPLETE',records=records,expected=len(rows)*len(variants),
        compressed_variants_run=any(v['method']=='round7_tail_key' for v in variants),
        paired_owner_checks=paired_checks,padded_visual_tokens=0,
        actual_visual_token_counts=visual_counts, actual_visual_token_counts_mod32=[n % 32 for n in visual_counts],
        full_native_complete_tokens_all_equal=True if native_indices else None,
        value_tail_padding_scope='physical final V bit word only; exact real visual token/position count preserved')
    (args.output/'complete.json').write_text(json.dumps(complete,indent=2))


if __name__=='__main__':
    main()
