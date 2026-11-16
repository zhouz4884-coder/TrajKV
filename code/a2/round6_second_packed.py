"""Independent LLaVA first-question packed width/depth E2E runner.

Uses the separately smoke-verified second-session HF bridge, actual PQ K2 /
whole-image binary V1, and the real two-stage packed reader. Nv is never padded.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_packed(model, processor, features, visual, args, variant, full_trace):
    import torch
    from round6_depth_bank import DepthResidualBank
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
        DepthResidualBank.settings = variant
        bank = DepthResidualBank(bank_layer_proxy(model), cache, visual, 'round6_depth',
            Path(args.legacy_scripts), Path(args.kivi_source), keep_reference=False)
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
        result = dict(variant=variant['name'], variant_config=variant, method='round6_depth',
            protocol='first_question_image_boundary_free_running',
            turns=[dict(turn=1,token_ids=ids,answer=processor.tokenizer.decode(ids,skip_special_tokens=True).strip())],
            visual_tokens=int(visual.numel()),padded_visual_tokens=0,owned_initial=initial_owner,
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
    sys.path.insert(0,str(Path(a.legacy_scripts)))
    torch.set_num_threads(4);torch.manual_seed(1212);torch.backends.cuda.matmul.allow_tf32=False
    rows = json.loads(Path(a.samples).read_text())
    rows = rows[int(config.get('start',0)):int(config.get('start',0))+int(config['count'])]
    if len(rows)!=int(config['count']) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Missing/duplicate second-model cases')
    variants = config['variants']
    full_indices = [i for i,v in enumerate(variants) if v['method']=='full']
    if len(full_indices)!=1 or any(v['method'] not in ('native_hf','full','round6_depth') for v in variants):
        raise ValueError('Require one Full reference and supported packed codecs only')
    first_compressed = min((i for i,v in enumerate(variants) if v['method']=='round6_depth'),default=len(variants))
    if full_indices[0] >= first_compressed:
        raise ValueError('Full reference must precede packed free generation')
    processor = AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    geometry = model_geometry(model)
    if (geometry['layers'],geometry['kv_heads'],geometry['head_dim']) != (32,32,128):
        raise ValueError('This static second-backbone configuration is specifically L32/Hkv32/D128')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    external = {v[key]:sha(v[key]) for v in variants for key in ('direction_book_path','metadata_factor_path') if v.get(key)}
    for variant in variants:
        if variant['method']!='round6_depth':
            continue
        factor = json.loads(Path(variant['metadata_factor_path']).read_text())
        if factor.get('geometry') != {'layers':32,'kv_heads':32,'head_dim':128} or factor.get('calibration_mode')!='identity_no_task_data':
            raise ValueError('Second-model factors must be newly declared 32x32 identities')
        if variant.get('v_group_size') != 'all':
            raise ValueError('This no-padding Nv576 run requires whole-image V metadata')
    manifest = dict(schema='trajkv.round6.second-backbone-packed.v1',config=config,config_sha256=sha(args.config),
        samples_sha256=sha(a.samples),selected_ids=[r['id'] for r in rows],geometry=geometry,
        model_config_sha256=sha(Path(a.model)/'config.json'),external_codec_inputs=external,
        answers_visible_to_runner=False,model_weights_frozen=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        sources={str(x.relative_to(ROOT)):sha(x) for x in [Path(__file__),Path(__file__).with_name('round6_second_session.py'),
            Path(__file__).with_name('round6_depth_bank.py'),Path(__file__).with_name('round6_depth_attention.py'),Path(__file__).with_name('round6_depth_codec.py')]},
        scope='unfitted second-backbone development transfer; actual packed E2E; no blind-confirmation claim')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records, paired_checks = 0, []
    with torch.inference_mode(), (args.output/'records.jsonl').open('x') as stream:
        for row in rows:
            features,visual,prompt = first_question_features(processor,model,row)
            if visual.numel()%32 or visual.numel()!=int(config.get('expected_visual_tokens',576)):
                raise ValueError('Unexpected actual visual token count; no padding is allowed')
            native,full,groups = None,None,{}
            for variant in variants:
                if variant['method']=='round6_depth':
                    result = run_packed(model,processor,features,visual,a,variant,full)
                    group = variant.get('axis_control','static')
                    current = (result['complete_visual_owner_bytes'],result['depth_residual_stages'],
                        result['depth_residual_K_stages'],result['depth_residual_V_stages'])
                    if group in groups and groups[group]!=current:
                        raise RuntimeError('Width/pair controls differ in actual complete bytes/stage axis costs')
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
                    split=row.get('split'),model=a.model,prompt_metadata=prompt)
                stream.write(json.dumps(result)+'\n');stream.flush();records+=1
                print(json.dumps({'done':records,'id':row['id'],'variant':variant['name'],'visual_tokens':int(visual.numel()),
                    'actual_bits':result['actual_visual_bits_including_encoder'],'answer':result['turns'][0]['answer']}),flush=True)
            # Axis-matched spread/random must also have the exact same owner.
            if len(set(groups.values()))>1:
                raise RuntimeError('Static spread/random controls differ in actual complete owner')
            paired_checks.append({'id':row['id'],'matched_bytes_and_axis_stage_costs':True,'groups':groups})
    for path,digest in external.items():
        if sha(path)!=digest:
            raise RuntimeError('Frozen second-backbone encoder input changed')
    complete=dict(status='COMPLETE',records=records,expected=len(rows)*len(variants),
        compressed_variants_run=any(v['method']=='round6_depth' for v in variants),
        paired_owner_checks=paired_checks,padded_visual_tokens=0)
    (args.output/'complete.json').write_text(json.dumps(complete,indent=2))


if __name__=='__main__':
    main()
