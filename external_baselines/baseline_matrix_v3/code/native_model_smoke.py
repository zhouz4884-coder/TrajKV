"""Native FP16 image/prefill/decode P0. No KV compression is performed here."""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from model_registry import cache_layers, get_model_spec, qwen_language_model, qwen_visual_positions


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for b in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def native_rows(job):
    rows = job.get('image_paths')
    if rows is None:
        manifest = Path(job['image_manifest'])
        raw = manifest.read_text()
        rows = json.loads(raw) if manifest.suffix == '.json' else [json.loads(x) for x in raw.splitlines() if x.strip()]
        if isinstance(rows, dict):
            rows = rows.get('items', rows.get('images', rows.get('records')))
    result = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            r = {'image_path': r}
        r = dict(r)
        p = r.get('image_path', r.get('image', r.get('path')))
        if not p:
            raise ValueError(f'no image path in manifest row {i}')
        r['image_path'] = p
        r.setdefault('source_id', str(i))
        result.append(r)
    return result[:int(job.get('limit', 10))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    a = ap.parse_args()
    job = json.loads(a.job.read_text())
    outdir = Path(job['output_dir'])
    if os.environ.get('BASELINE_DATA_ROOT'):
        allowed_root = Path(os.environ['BASELINE_DATA_ROOT']).resolve()
        if not outdir.resolve().is_relative_to(allowed_root):
            raise ValueError('Output must remain inside the launcher baseline project root')
    outdir.mkdir(parents=True, exist_ok=False)
    marker = outdir / 'marker.json'
    spec = get_model_spec(job['model_id'])
    is_qwen = spec['native_adapter'] in ('qwen2_5_vl', 'qwen3_vl')
    info = {'status': 'RUNNING', 'model_spec': spec, 'job': job,
            'argv': sys.argv, 'pid': os.getpid(), 'python': sys.executable,
            'started_unix': time.time(), 'source_sha256': sha(__file__),
            'registry_sha256': sha(Path(__file__).with_name('model_registry.py'))}
    marker.write_text(json.dumps(info, indent=2))
    records = []
    active_row = None
    try:
        if is_qwen:
            integrity_path = Path(job['download_integrity_report'])
            integrity = json.loads(integrity_path.read_text())
            if integrity.get('status') != 'PASS' or integrity.get('revision') != spec['revision']:
                raise RuntimeError('checkpoint integrity gate has not passed for locked revision')
            info['download_integrity_report_sha256'] = sha(integrity_path)
            lock_path = Path(job.get('hf_revision_lock', str(Path(job['model_path']).parents[3] / 'configs/models/hf_revision_lock.json')))
            lock = json.loads(lock_path.read_text())[spec['hf_id']]
            if lock['sha'] != spec['revision']:
                raise RuntimeError('model metadata lock revision mismatch')
            checked_metadata = []
            for entry in lock['siblings']:
                name = entry['rfilename']
                if '/' in name or not name.endswith(('.json','.txt','.jinja','.model')):
                    continue
                raw = (Path(job['model_path']) / name).read_bytes()
                lfs = entry.get('lfs')
                actual = hashlib.sha256(raw).hexdigest() if lfs else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
                expected = lfs['sha256'] if lfs else entry['blobId']
                if len(raw) != entry['size'] or actual != expected:
                    raise RuntimeError('official processor/config metadata integrity mismatch: '+name)
                checked_metadata.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),
                    'official_identity_kind':'LFS_SHA256' if lfs else 'GIT_BLOB_SHA1','exact_match':True})
            info['metadata_integrity'] = checked_metadata
            info['model_revision_lock_sha256'] = sha(lock_path)
        else:
            local_model_path = Path(job['model_path'])
            index = json.loads((local_model_path / 'model.safetensors.index.json').read_text())
            required = sorted(set(index['weight_map'].values()))
            info['checkpoint_identity_status'] = 'LOCAL_EXACT_FILE_SHA256_OFFICIAL_REVISION_NOT_ESTABLISHED'
            info['checkpoint_files'] = [{'file': f, 'bytes': (local_model_path/f).stat().st_size,
                'sha256': sha(local_model_path/f)} for f in required + ['config.json','model.safetensors.index.json']]
            info['hub_download_metadata_present'] = (local_model_path/'.cache/huggingface/download').exists()
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
        torch.set_grad_enabled(False)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        info['environment'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'hardware': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
            'python': platform.python_version(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cache_env': {k: os.environ.get(k) for k in ('HF_HOME', 'TMPDIR', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH')}}
        model_path = Path(job['model_path'])
        info['model_config_sha256'] = sha(model_path / 'config.json')
        info['native_weight_dtype'] = json.loads((model_path / 'config.json').read_text()).get('torch_dtype',
            json.loads((model_path / 'config.json').read_text()).get('text_config', {}).get('dtype'))
        info['dtype_override'] = 'float16'
        torch.cuda.reset_peak_memory_stats()
        load0 = time.perf_counter()
        processor_kwargs = {}
        if is_qwen:
            processor_kwargs.update(min_pixels=int(job.get('min_pixels', spec['processor_spatial_factor'] ** 2 * 4)),
                max_pixels=int(job.get('max_pixels', 448 * 448)))
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, **processor_kwargs)
        info['processor_image_config'] = processor.image_processor.to_dict()
        info['processor_source_config_sha256'] = sha(model_path / 'preprocessor_config.json')
        model = AutoModelForImageTextToText.from_pretrained(str(model_path),
            dtype=torch.float16, device_map={'': 0}, local_files_only=True,
            attn_implementation=job.get('attn_implementation', 'sdpa')).eval()
        torch.cuda.synchronize()
        info['load_ms'] = (time.perf_counter() - load0) * 1000
        info['actual_architecture'] = type(model).__name__
        info['native_generation_config'] = model.generation_config.to_dict()
        info['generation_overrides'] = {'do_sample': False, 'max_new_tokens': int(job.get('max_new_tokens', 32))}
        info['load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        parameter_owners = {}
        for p in model.parameters():
            st = p.untyped_storage()
            parameter_owners[(st.data_ptr(), st.nbytes())] = st.nbytes()
            if p.dtype != torch.float16:
                raise RuntimeError(f'unexpected weight dtype {p.dtype}')
        info['weights_resident_unique_storage_bytes'] = sum(parameter_owners.values())
        module = sys.modules[type(model).__module__]
        info['transformers_model_source'] = {'file_name': Path(module.__file__).name, 'sha256': sha(module.__file__)}
        native_source = outdir / Path(module.__file__).name
        native_source.write_bytes(Path(module.__file__).read_bytes())
        marker.write_text(json.dumps(info, indent=2))
        snapshots = []
        def capture_native(module, args, kwargs):
            snap = {}
            pos = kwargs.get('position_ids')
            if pos is not None:
                snap['position_ids'] = pos.detach().cpu().tolist()
                snap['position_ids_shape'] = list(pos.shape)
            deep = kwargs.get('deepstack_visual_embeds')
            snap['deepstack_shapes'] = [list(x.shape) for x in deep] if deep is not None else []
            mask = kwargs.get('visual_pos_masks')
            if mask is not None:
                snap['visual_pos_mask_count'] = int(mask.sum())
            snapshots.append(snap)
        handle = qwen_language_model(model).register_forward_pre_hook(capture_native, with_kwargs=True)
        for i, source in enumerate(native_rows(job)):
            image_path = Path(source['image_path'])
            image = Image.open(image_path).convert('RGB')
            active_row = {'status': 'RUNNING', 'index': i, 'source_id': source['source_id'],
                'original_image_sha256': sha(image_path), 'original_dimensions': list(image.size),
                'image_path': str(image_path), 'model_id': job['model_id'],
                'cap_or_max_pixels': int(job.get('max_pixels', 448 * 448))}
            expected_image_sha = source.get('source_image_sha256', source.get('image_sha256'))
            if expected_image_sha and expected_image_sha != active_row['original_image_sha256']:
                raise RuntimeError('source image hash differs from frozen P0 manifest')
            question = source.get('question', job.get('question', 'Describe the image briefly.'))
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                        {'type': 'text', 'text': question}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            call_kwargs = {} if is_qwen else {'crop_to_patches': True, 'max_patches': int(job.get('max_patches',8))}
            inputs = processor(text=[prompt], images=[image], return_tensors='pt', **call_kwargs).to(device)
            if is_qwen:
                rope_signature = inspect.signature(model.model.get_rope_index).parameters
                rope_kwargs = {k: v for k, v in inputs.items() if k in rope_signature}
                native_rope_positions, native_rope_deltas = model.model.get_rope_index(**rope_kwargs)
            else:
                native_rope_positions = torch.arange(inputs['input_ids'].shape[1],device=device)[None]
                native_rope_deltas = None
            visual = qwen_visual_positions(inputs['input_ids'], spec)
            n = inputs['input_ids'].shape[1]
            resolution_audit = {}
            if is_qwen:
                grid = inputs['image_grid_thw'][0].detach().cpu().tolist()
                expected_visual = grid[0] * grid[1] * grid[2] // 4
                patch_width = 14 if job['model_id'] == 'M2' else 16
                actual_pixels = grid[1] * grid[2] * patch_width ** 2
                resolution_audit = {'expected_visual_tokens_from_grid': expected_visual,
                    'processed_spatial_pixels': actual_pixels}
                if len(visual) != expected_visual or actual_pixels > int(job.get('max_pixels',448*448)):
                    raise RuntimeError('native processor grid violates declared visual-token/pixel contract')
            elif len(visual) % 256 or len(visual) > (int(job.get('max_patches',8))+1)*256:
                raise RuntimeError('native InternVL patch/token geometry violates cap+thumbnail contract')
            row = {'status': 'RUNNING', 'index': i, 'source_id': source['source_id'],
                'original_image_sha256': sha(image_path), 'original_dimensions': list(image.size),
                'image_path': str(image_path), 'question': question, 'full_prompt': prompt,
                'full_prompt_hash': hash_text(prompt), 'chat_template_hash': hash_text(processor.chat_template),
                'actual_visual_token_count': len(visual), 'visual_positions': visual.cpu().tolist(),
                'input_token_ids': inputs['input_ids'][0].detach().cpu().tolist(),
                'native_modality_ids': inputs['mm_token_type_ids'][0].detach().cpu().tolist() if 'mm_token_type_ids' in inputs else None,
                'prospective_visual_prefix_boundary': int(visual[-1])+1,
                'question_suffix_token_ids': inputs['input_ids'][0,int(visual[-1])+1:].detach().cpu().tolist(),
                'question_suffix_sha256': hash_text(json.dumps(inputs['input_ids'][0,int(visual[-1])+1:].detach().cpu().tolist(),separators=(',',':'))),
                'compression_boundary': 'NONE_NATIVE_FULL_FORWARD_P0',
                'image_grid_thw': inputs['image_grid_thw'].cpu().tolist() if 'image_grid_thw' in inputs else None,
                'pixel_values_shape': list(inputs['pixel_values'].shape),
                'input_tokens': n, 'model_id': job['model_id'], 'model_commit': spec['revision'],
                'dtype': 'float16', 'processor_version': transformers.__version__,
                'cap_or_max_pixels': int(job.get('max_pixels', 448 * 448)) if is_qwen else int(job.get('max_patches',8))}
            row.update(resolution_audit)
            active_row = row
            row['native_get_rope_index' if is_qwen else 'native_serial_position_contract'] = {
                'position_ids': native_rope_positions.detach().cpu().tolist(),
                'position_ids_shape': list(native_rope_positions.shape),
                'rope_deltas': native_rope_deltas.detach().cpu().tolist() if native_rope_deltas is not None else None}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            snapshots.clear()
            t0 = time.perf_counter()
            prefill_kwargs = {} if is_qwen else {'position_ids': native_rope_positions}
            output = model(**inputs, **prefill_kwargs, use_cache=True, return_dict=True, logits_to_keep=1)
            torch.cuda.synchronize()
            row['native_prefill_ms'] = (time.perf_counter() - t0) * 1000
            row['native_position_snapshot'] = snapshots[0]
            row['rope_deltas'] = model.model.rope_deltas.detach().cpu().tolist() if is_qwen and model.model.rope_deltas is not None else None
            kv = list(cache_layers(output.past_key_values))
            row['cache_class'] = type(output.past_key_values).__name__
            row['cache_layers'] = [{'k_shape': list(k.shape), 'v_shape': list(v.shape),
                'k_dtype': str(k.dtype), 'v_dtype': str(v.dtype),
                'k_bytes': k.numel() * k.element_size(), 'v_bytes': v.numel() * v.element_size()}
                for k, v in kv]
            for k, v in kv:
                expected = (1, spec['kv_heads'], n, spec['head_dim'])
                if tuple(k.shape) != expected or tuple(v.shape) != expected:
                    raise RuntimeError(f'cache geometry mismatch expected={expected}, observed={k.shape},{v.shape}')
            if len(kv) != spec['layers']:
                raise RuntimeError('cache layer count mismatch')
            if spec['deepstack_visual_indexes'] and len(snapshots[0]['deepstack_shapes']) != 3:
                raise RuntimeError('native Qwen3 DeepStack missing')
            row['full_kv_bytes'] = sum(x['k_bytes'] + x['v_bytes'] for x in row['cache_layers'])
            row['visual_kv_bytes'] = len(visual) * spec['layers'] * spec['kv_heads'] * spec['head_dim'] * 2 * 2
            row['raw_logit_argmax_first_token_id'] = int(output.logits[:, -1:].argmax(-1).item())
            first_scores = output.logits[:, -1, :].float().clone()
            repetition_value = model.generation_config.repetition_penalty
            repetition_penalty = float(repetition_value if repetition_value is not None else 1.0)
            if repetition_penalty != 1.0:
                first_scores = RepetitionPenaltyLogitsProcessor(repetition_penalty)(inputs['input_ids'], first_scores)
            next_token = first_scores.argmax(-1, keepdim=True)
            row['native_repetition_penalty'] = repetition_penalty
            row['manual_first_token_id'] = int(next_token.item())
            t1 = time.perf_counter()
            decode_positions = native_rope_positions[..., -1:] + 1
            row['first_decode_position_ids'] = decode_positions.cpu().tolist()
            decode = model(input_ids=next_token,
                attention_mask=torch.ones((1, n + 1), dtype=inputs['attention_mask'].dtype, device=device),
                position_ids=decode_positions,
                past_key_values=output.past_key_values, use_cache=True, return_dict=True, logits_to_keep=1)
            torch.cuda.synchronize()
            row['native_first_decode_ms'] = (time.perf_counter() - t1) * 1000
            row['cache_length_after_decode'] = int(decode.past_key_values.get_seq_length())
            row['first_decode_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            del decode, output, kv
            gc.collect()
            torch.cuda.empty_cache()
            t2 = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=int(job.get('max_new_tokens', 32)),
                do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            row['native_generate_ms'] = (time.perf_counter() - t2) * 1000
            ids = generated[0, n:].cpu().tolist()
            row['output_token_ids'] = ids
            row['raw_prediction'] = processor.decode(ids, skip_special_tokens=True)
            row['first_token_matches_native_generate'] = bool(ids and ids[0] == row['manual_first_token_id'])
            if not row['first_token_matches_native_generate']:
                raise RuntimeError('manual first token differs from native greedy generate')
            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            row['status'] = 'PASS'
            with (outdir / 'records.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            records.append(row)
            print(json.dumps({'event': 'native_p0_case', 'model_id': job['model_id'], 'index': i,
                'status': row['status'], 'tokens': len(visual), 'answer': row['raw_prediction'],
                'peak_allocated_bytes': row['peak_allocated_bytes']}), flush=True)
            del inputs, generated, next_token, image
            gc.collect()
        handle.remove()
        if len(records) != int(job.get('limit', 10)):
            raise RuntimeError('insufficient P0 images')
        if len(set(tuple(x['original_dimensions']) for x in records)) != len(records):
            raise RuntimeError('P0 requires distinct image dimensions')
        info.update(status='PASS', completed_cases=len(records),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
            completed_unix=time.time())
    except Exception as exc:
        oom = 'out of memory' in str(exc).lower()
        info.update(status='OOM_AT_DECLARED_INPUT' if oom else 'FAILED_RUN',
            error=repr(exc), traceback=traceback.format_exc(),
            completed_cases=len(records), completed_unix=time.time())
        if active_row:
            active_row.update(status=info['status'], error_type=type(exc).__name__)
            try:
                active_row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                active_row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            except Exception:
                pass
            (outdir / 'failed_case.json').write_text(json.dumps(active_row, indent=2))
        (outdir / 'exception.txt').write_text(info['traceback'])
        print(info['traceback'], flush=True)
        raise
    finally:
        marker.write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
