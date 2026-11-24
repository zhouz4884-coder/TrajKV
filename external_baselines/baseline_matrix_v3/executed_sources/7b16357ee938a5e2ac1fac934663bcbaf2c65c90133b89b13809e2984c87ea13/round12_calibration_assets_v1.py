"""Actual CUDA, no-answer native-prefix calibration and small global books.

This is QA0 calibration, never free-answer task evaluation. All inputs are the
previously frozen CAL fit/select cohorts, disjoint from R12 development.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def balanced(rows, per_domain):
    result = []
    for domain in ('textvqa', 'infovqa', 'docvqa', 'chartqa'):
        chosen = [r for r in rows if r['dataset'] == domain][:per_domain]
        assert len(chosen) == per_domain
        result.extend(chosen)
    return result


def lloyd(points, words, iterations, generator, chunk=8192):
    import torch
    order = torch.randperm(len(points), device=points.device, generator=generator)[:words]
    centers = points[order].clone().float()
    losses = []
    for it in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(words, device=points.device, dtype=torch.float32)
        error = torch.zeros((), device=points.device, dtype=torch.float64)
        c2 = centers.square().sum(-1)[None]
        for start in range(0, len(points), chunk):
            x = points[start:start + chunk]
            d = x.square().sum(-1, keepdim=True) + c2 - 2 * (x @ centers.T)
            value, ids = d.min(-1)
            sums.index_add_(0, ids, x)
            counts.index_add_(0, ids, torch.ones(len(x), device=x.device))
            error += value.double().sum()
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]
        if not bool(occupied.all()):
            number = int((~occupied).sum())
            centers[~occupied] = points[torch.randint(len(points), (number,), device=points.device, generator=generator)]
        loss = float(error / len(points))
        losses.append(loss)
        print(json.dumps({'event': 'LLOYD', 'words': words, 'dim': points.shape[-1],
                          'iteration': it + 1, 'MSE_per_block': loss,
                          'occupied': int(occupied.sum())}), flush=True)
    return centers.half().contiguous(), losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text(encoding='utf-8-sig'))
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from m1_protocol import prepare_m1_inputs, m1_prefix_forward
    from evaluate import model_identity
    from model_registry import get_model_spec
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(20261124)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6)
    for key in ('cal_fit_inputs', 'cal_select_inputs'):
        assert sha(c[key]) == c[key + '_sha256']
    fit_all = [json.loads(x) for x in Path(c['cal_fit_inputs']).read_text().splitlines() if x.strip()]
    sel_all = [json.loads(x) for x in Path(c['cal_select_inputs']).read_text().splitlines() if x.strip()]
    fit, holdout = balanced(fit_all, 2), balanced(sel_all, 1)
    forbidden = set(c['development_image_sha256'])
    assert not forbidden & {x['original_image_sha256'] for x in fit + holdout}
    assert not {x['original_image_sha256'] for x in fit} & {x['original_image_sha256'] for x in holdout}
    assert not any(set(r) & {'answers', 'refs', 'labels', 'reference_answers', 'scores'} for r in fit + holdout)
    identity = model_identity(c, get_model_spec('M1'))
    processor = AutoProcessor.from_pretrained(c['model_path'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(c['model_path'], dtype=torch.float16,
        device_map={'': 0}, local_files_only=True, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    parameter_count = sum(x.numel() for x in model.parameters())
    assert parameter_count == 7944373760 and all(x.dtype == torch.float16 for x in model.parameters())
    sys.path.insert(0, c['legacy_a2'])
    from fast_wht import wht
    kparts, vparts, source_events = {'fit': [], 'holdout': []}, {'fit': [], 'holdout': []}, []
    with torch.no_grad():
        for role, rows in (('fit', fit), ('holdout', holdout)):
            for row in rows:
                assert sha(row['image_path']) == row['original_image_sha256']
                with Image.open(row['image_path']) as im:
                    image = im.convert('RGB')
                batch = prepare_m1_inputs(processor, image, row['question'] + c['question_suffix'], max_patches=8)
                out = m1_prefix_forward(model, batch)
                cache = out.past_key_values
                n = int(batch.visual_positions.numel())
                for layer in cache.layers:
                    key = layer.keys.index_select(2, batch.visual_positions).float()
                    mean = key.mean(-2, keepdim=True).half()
                    z = wht(key - mean.float())
                    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-20)
                    x = z * (128 ** .5 / norm)
                    kparts[role].append(x[:, :, ::4].reshape(-1, 128).half().cpu())
                    value = layer.values.index_select(2, batch.visual_positions).float()
                    mu = value.mean(-2, keepdim=True)
                    std = value.std(-2, unbiased=False, keepdim=True).clamp_min(1e-8)
                    vparts[role].append(((value - mu) / std)[:, :, ::4].flatten().half().cpu())
                    del key, mean, z, norm, x, value, mu, std
                source_events.append({'id': row['id'], 'role': role, 'image_sha256': row['original_image_sha256'],
                                      'visual_tokens': n, 'native_prefix_model_forwards': 1})
                print(json.dumps({'event': 'NATIVE_CAL_PREFIX', **source_events[-1]}), flush=True)
                del out, cache, batch, image
                gc.collect()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    # No model, native KV or answers remain while fitting the global codebooks.
    vx = torch.cat(vparts['fit']).float().cuda()
    ax = vx.abs()
    inner, outer = .45278, 1.5104
    for _ in range(24):
        cut = .5 * (inner + outer)
        inside = ax <= cut
        inner = float(ax[inside].mean())
        outer = float(ax[~inside].mean())
    ratio = {'schema': 'trajkv.round12.value_ratio.v1', 'levels': [-outer, -inner, inner, outer],
             'fit_scope': 'balanced_cal_fit8_actual_native_prefix_channel_standardized_V',
             'source_events': source_events, 'answers_used': False, 'fit_method': 'symmetric_four_level_Lloyd_24',
             'deployed_affine_Half_and_f0_fit_is_per_image': True, 'parameter_bytes': 8}
    write(a.output / 'value_ratio.json', ratio)
    print(json.dumps({'event': 'RATIO_COMPLETE', 'levels': ratio['levels'],
                      'sha256': sha(a.output / 'value_ratio.json')}), flush=True)
    del vx, ax, vparts
    gc.collect()
    train = torch.cat(kparts['fit']).float().cuda()
    held = torch.cat(kparts['holdout']).half()
    torch.save({'normalized_K': held, 'source_events': [e for e in source_events if e['role'] == 'holdout']},
               a.output / 'heldout_K.pt')
    generator = torch.Generator(device='cuda').manual_seed(20261124)
    assets = {}
    for name, words, dim in (('g4_1024', 1024, 4), ('g8_4096', 4096, 8)):
        allparts = train.reshape(-1, dim)
        ids = torch.randint(len(allparts), (int(c.get('train_blocks', 1500000)),),
                            device='cuda', generator=generator)
        points = allparts[ids].contiguous()
        centers, loss = lloyd(points, words, int(c.get('iterations', 24)), generator,
                              chunk=int(c.get('chunk', 8192)))
        payload = {'schema': 'trajkv.direction-book.v1', 'dimension': dim, 'centroids': words,
                   'nominal_bits_per_scalar': (words - 1).bit_length() / dim,
                   'centers': centers.float().cpu().tolist(), 'storage_dtype': 'float16',
                   'actual_book_bytes': centers.numel() * 2, 'answers_used': False,
                   'model_id': 'M1', 'model_revision': c['model_revision'],
                   'calibration_config_sha256': sha(a.config), 'source_events': source_events,
                   'fit_blocks': len(points), 'iterations': len(loss), 'training_loss': loss,
                   'training_is_GPU_tensor_calibration_not_task_QA': True}
        target = a.output / (name + '.json')
        write(target, payload)
        assets[name] = {'path': str(target), 'sha256': sha(target), 'book_bytes': centers.numel() * 2}
        print(json.dumps({'event': 'BOOK_COMPLETE', 'name': name, **assets[name]}), flush=True)
        del points, centers, ids
        torch.cuda.empty_cache()
    write(a.output / 'complete.json', {'status': 'PASS_CALIBRATION_ASSETS', 'new_free_QA': 0,
          'native_prefix_model_forwards': len(source_events), 'model_parameter_count': parameter_count,
          'model_identity': identity, 'assets': assets, 'ratio_sha256': sha(a.output / 'value_ratio.json'),
          'sources': source_events, 'development_overlap': 0, 'config_sha256': sha(a.config),
          'wall_seconds': time.time() - started, 'hardware': torch.cuda.get_device_name(),
          'torch_version': torch.__version__})


if __name__ == '__main__':
    main()
