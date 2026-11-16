"""First-question-only, model-specific features and generic HF attention bridge.

No InternVL continuation suffix is used. Existing banks receive a lightweight
layer-path proxy only; the real model and frozen legacy sources stay unchanged.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch


def decoder_layers(model):
    for path in ('model.language_model.layers', 'model.language_model.model.layers',
                 'language_model.model.layers', 'language_model.layers',
                 'model.layers'):
        value = model
        try:
            for part in path.split('.'):
                value = getattr(value, part)
        except AttributeError:
            continue
        if len(value) and all(hasattr(layer, 'self_attn') for layer in value):
            return list(value), path
    raise ValueError('Unsupported decoder layer path')


def bank_layer_proxy(model):
    layers, _ = decoder_layers(model)
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))


def image_token_id(model):
    config = model.config
    value = getattr(config, 'image_token_id', None)
    alias = getattr(config, 'image_token_index', None)
    if value is not None and alias is not None and int(value) != int(alias):
        raise ValueError('Conflicting image token aliases')
    token = value if value is not None else alias
    if token is None:
        raise ValueError('Model has no image token ID/index')
    return int(token)


def model_geometry(model):
    layers, path = decoder_layers(model)
    config = getattr(model.config, 'text_config', None) or model.config
    heads = int(config.num_attention_heads)
    kv = int(getattr(config, 'num_key_value_heads', heads))
    dimension = int(getattr(config, 'head_dim', None) or int(config.hidden_size) // heads)
    if len(layers) != int(config.num_hidden_layers) or heads % kv:
        raise ValueError('Invalid actual decoder geometry')
    if any(layer.self_attn.q_proj.out_features != heads * dimension or
           layer.self_attn.k_proj.out_features != kv * dimension or
           layer.self_attn.v_proj.out_features != kv * dimension for layer in layers):
        raise ValueError('Config heads disagree with actual attention projection sizes')
    return dict(decoder_layer_path=path, layers=len(layers), query_heads=heads,
                kv_heads=kv, head_dim=dimension, hidden_size=int(config.hidden_size),
                image_token_id=image_token_id(model), model_type=model.config.model_type)


def first_question_features(processor, model, row):
    from PIL import Image
    if model.config.model_type not in ('llava', 'llava_next'):
        raise ValueError('This first smoke adapter is restricted to HF LLaVA families')
    if any(key in row for key in ('answer', 'answers', 'references')):
        raise ValueError('Answer-bearing inference runner')
    names = row.get('remote_images', row.get('images', []))
    if len(names) != 1 or not row.get('questions'):
        raise ValueError('Exactly one image and a first question required')
    with Image.open(names[0]) as im:
        image = im.convert('RGB')
    question = row['questions'][0]
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
    template = getattr(processor, 'chat_template', None) or getattr(processor.tokenizer, 'chat_template', None)
    if template:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        protocol = 'locally installed official processor/tokenizer chat template'
    else:
        # Same first-turn serialization as the pinned llava-1.5 official
        # template; never substitute an InternVL multiturm continuation.
        if model.config.model_type != 'llava':
            raise ValueError('llava_next requires its official local chat template')
        prompt = 'USER: <image>\n' + question + ' ASSISTANT:'
        protocol = 'llava-1.5 official first-turn USER/image/ASSISTANT serialization fallback'
    # LLaVA processors expand exactly the real vision tokens. No crop-to-patches
    # InternVL kwargs and no padding to a codec group size are passed.
    features = processor(text=[prompt], images=[image], return_tensors='pt')
    token = image_token_id(model)
    visual = torch.nonzero(features['input_ids'][0] == token).flatten()
    if visual.numel() == 0 or not bool((visual[1:] == visual[:-1] + 1).all()):
        raise ValueError('One contiguous real image-token segment required')
    return features, visual, dict(prompt_protocol=protocol, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_template_sha256=hashlib.sha256(str(template).encode()).hexdigest() if template else None,
        visual_tokens=int(visual.numel()), padded_visual_tokens=0, turns=1,
        input_sequence_length=int(features['input_ids'].shape[-1]))


@contextmanager
def bank_attention(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    layers, _ = decoder_layers(model)
    modules = {id(layer.self_attn) for layer in layers}

    def apply(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if id(module) not in modules:
            return original(module, query, key, value, attention_mask,
                            dropout=dropout, scaling=scaling, **kwargs)
        if not hasattr(module, '_p185_read'):
            raise RuntimeError('Missing segmented-bank hook')
        if dropout or (scaling is not None and abs(float(scaling) - query.shape[-1] ** -.5) > 1e-8):
            raise ValueError('Bank requires no dropout and the standard head-dimension scale')
        output = module._p185_read(query)
        return output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register('sdpa', apply)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


def forward(model, **kwargs):
    parameters = inspect.signature(model.forward).parameters
    if 'logits_to_keep' in parameters:
        kwargs['logits_to_keep'] = 1
    elif 'num_logits_to_keep' in parameters:
        kwargs['num_logits_to_keep'] = 1
    return model(**kwargs, use_cache=True, return_dict=True)


def consume(model, cache, ids, logical):
    out = forward(model, input_ids=ids, past_key_values=cache,
        position_ids=torch.arange(logical, logical + ids.shape[-1], device=ids.device)[None])
    return out, logical + ids.shape[-1]


def stop_ids(model, processor):
    result = set()
    for value in (model.generation_config.eos_token_id, processor.tokenizer.eos_token_id):
        if value is not None:
            result.update(value if isinstance(value, (list, tuple)) else [value])
    return {int(x) for x in result}


def run_first_question(model, processor, features, visual, args, variant, native_trace=None):
    """Run free generation with native HF or one existing/new compatible bank.

    native_trace is an in-memory earlier native run; it supplies same-prefix
    numerical comparisons, never forced tokens or selection signals.
    """
    import gc
    import time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    inputs = {key: value.to(model.device, dtype=model.dtype if value.is_floating_point() else value.dtype)
              for key, value in features.items()}
    visual = visual.to(model.device)
    out = forward(model, **inputs)
    cache, logits = out.past_key_values, out.logits[:, -1].clone()
    logical = inputs['input_ids'].shape[-1]
    bank = None
    try:
        if variant['method'] != 'native_hf':
            from masked_bank import A2Bank
            if variant['method'] == 'full':
                cls = A2Bank
            elif variant['method'] == 'direction_meta':
                from metadata_direction_bank import MetadataDirectionBank
                cls = MetadataDirectionBank
                if variant.get('metadata_factor_path'):
                    raise ValueError('Second-model factors must be independently fitted before use')
            else:
                raise ValueError('Smoke enables native, Full and identity-metadata PQbase only')
            geometry = model_geometry(model)
            if geometry['head_dim'] != 128:
                raise ValueError('Existing Full/PQ kernel smoke requires actual D128')
            cls.settings = variant
            bank = cls(bank_layer_proxy(model), cache, visual, variant['method'],
                       Path(args.legacy_scripts), Path(args.kivi_source), keep_reference=False)
            boundary = int(visual[-1].item()) + 1
            suffix = inputs['input_ids'][:, boundary:].clone()
            if suffix.shape[-1] == 0:
                raise ValueError('First question/assistant suffix is empty')
            for state in bank.states:
                state.text_length = int((state.text_pos[:state.text_length] < boundary).sum().item())
                state.length = boundary
            bank.turn, bank.phase, bank.pass_kind = 1, 'question_prefill', 'committed'
            bank.refine_layers = set()
            with bank_attention(model):
                out, logical = consume(model, cache, suffix, boundary)
            logits = out.logits[:, -1].clone()
            initial_owner = bank.owned()
            bank.on_turn_begin(1, logical)
        else:
            if not hasattr(cache, 'layers'):
                raise ValueError('Expected modern HF cache.layers for this smoke protocol')
            visual_bytes = sum(layer.keys.index_select(2, visual).numel() * layer.keys.element_size()
                               + layer.values.index_select(2, visual).numel() * layer.values.element_size()
                               for layer in cache.layers)
            initial_owner = dict(visual_base=visual_bytes, visual_residual=0, selector_metadata=0,
                                 visual_position=8 * visual.numel(), visual_fp16_denominator=visual_bytes,
                                 ownership_scope='native visual slice payload; combined native KV owner retained')
        del out, inputs
        ids, trace, comparisons = [], [], []
        stops = stop_ids(model, processor)
        prefix_matches = True
        for step in range(args.max_tokens):
            cpu_logits = logits.detach().float().cpu()
            token = int(logits.argmax(-1).item())
            trace.append(cpu_logits)
            if native_trace is not None and prefix_matches and step < len(native_trace['logits']):
                reference = native_trace['logits'][step]
                error = cpu_logits - reference
                comparisons.append(dict(step=step, max_abs_error=float(error.abs().max()),
                    relative_l2=float(error.norm() / reference.norm().clamp_min(1e-20)),
                    argmax_same=token == native_trace['token_ids'][step],
                    reference_argmax_margin=float((reference.topk(2).values[..., 0] - reference.topk(2).values[..., 1]).item())))
                prefix_matches = token == native_trace['token_ids'][step]
            ids.append(token)
            if token in stops or step + 1 == args.max_tokens:
                break
            incoming = torch.tensor([[token]], device=model.device)
            if bank is not None:
                bank.phase = 'decode'
            with bank_attention(model) if bank is not None else nullcontext():
                out, logical = consume(model, cache, incoming, logical)
            logits = out.logits[:, -1].clone()
            del out
        torch.cuda.synchronize()
        owner = initial_owner
        encoder = int(getattr(bank, 'encoder_parameter_bytes', 0)) if bank is not None else 0
        complete_visual_owner = sum(int(owner.get(key, 0)) for key in ('visual_base', 'visual_residual', 'selector_metadata', 'visual_position')) + encoder
        result = dict(variant=variant['name'], variant_config=variant,
            method=variant['method'], protocol='first_question_image_boundary_free_running' if bank else 'native_hf_first_question_free_running',
            turns=[dict(turn=1, token_ids=ids, answer=processor.tokenizer.decode(ids, skip_special_tokens=True).strip())],
            visual_tokens=int(visual.numel()), padded_visual_tokens=0,
            owned_initial=owner, encoder_calibration_parameters_bytes=encoder,
            actual_visual_bits_including_encoder=16 * complete_visual_owner / owner['visual_fp16_denominator'],
            initial_logits_sha256=hashlib.sha256(trace[0].numpy().tobytes()).hexdigest(),
            peak_allocated=torch.cuda.max_memory_allocated(), session_ms=1000 * (time.perf_counter() - started),
            native_same_prefix_logit_comparisons=comparisons,
            native_complete_tokens_equal=ids == native_trace['token_ids'] if native_trace else None,
            native_comparison_scope='free-running outputs; per-step logit comparisons stop once input token prefixes diverge',
            task_accuracy_eligible=True, answers_visible_to_runner=False)
        if bank is not None:
            bank.on_turn_end(1, logical)
            result.update(bank.result_metadata())
            result['read_events'] = bank.event_rows()
            result['read_ledger_exact'] = sum(e.get('selection_bytes', 0) + e.get('residual_bytes', 0)
                for e in result['read_events']) == bank.extra_read_bytes
            result['lowbit_deployment_candidate'] = variant['method'] != 'full'
        return result, dict(logits=trace, token_ids=ids)
    finally:
        if bank is not None:
            bank.close()
        del bank, cache
        gc.collect()
