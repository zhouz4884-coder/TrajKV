"""Bounded native-coordinate Round8 calibration collection.

The full-image mean, binary V partition, folded frame and old fields are formed
before sampling tokens.  These diagnostic source tensors are never deployment
owners.  This module launches nothing when imported.  Use the data-disk launcher
for the explicit model-running CLI; outputs are new, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


SCHEMA = 'trajkv.round8.native-dump.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(row):
    return str(row.get('source_id', row.get('source_image_id', row['image_id'])))


def unpack_binary(code, tokens):
    shifts = torch.arange(32, device=code.device, dtype=torch.int64)
    return ((code.long()[..., None] >> shifts) & 1).flatten(-2)[..., :tokens].byte()


def token_sample(tokens, maximum, source_id, seed):
    if maximum <= 0:
        raise ValueError('Positive maximum sampled tokens required')
    if tokens <= maximum:
        return torch.arange(tokens)
    salt = int(hashlib.sha256((str(seed) + source_id).encode()).hexdigest()[:16], 16)
    return torch.randperm(tokens, generator=torch.Generator().manual_seed(salt))[:maximum].sort().values


@torch.no_grad()
def collect_native_image(cache, visual, row, old_book, frozen_factors,
                         selected_stages, *, max_tokens=128, seed=8):
    """Return one image shard, preserving the frozen Round7 all-image frame.

    ``cache`` is native cache at the image/question boundary, with visual states
    still present. ``old_book`` must be the shipped 16x4 FP16 encoder book.
    No new reader, source reconstruction, answer, gradient or train/select
    decision is used to define these old fields.
    """
    # CUDA-only modules are imported only here so metadata/fit tools run on CPU.
    from fast_wht import wht
    from quantizers import binary_centroids
    from affine_rate_bank import fold
    from fused_progressive import decode_last_generic
    from round7_key_codec import encode_key, decode_key, unpack_nibbles
    from round6_depth_codec import encode_sign_depth, validate_stages

    if tuple(old_book.shape) != (16, 4) or old_book.dtype != torch.float16:
        raise ValueError('The frozen global16x4 FP16 encoder book is required')
    nv = int(visual.numel())
    if nv <= 0 or nv % 32:
        raise ValueError('Complete nonempty visual bit words required')
    nl = len(cache.layers)
    nh = cache.layers[0].keys.shape[1]
    stages = validate_stages(selected_stages, nl, nh)
    if len(stages) != 43 or sum(x[0] == 'k' for x in stages) != 23 or sum(x[0] == 'v' for x in stages) != 20:
        raise ValueError('Round8 first batch requires the frozen 23K20V43 stage mask')
    if frozen_factors.get('value_group_size') != 'all':
        raise ValueError('Main Round8 collection requires the frozen V all-image frame')
    device = visual.device
    params = {key: torch.tensor(frozen_factors[key], device=device, dtype=torch.float32)
              for key in ('key_head_factors', 'value_head_factors', 'value_center_offsets')
              if frozen_factors.get(key) is not None}
    for key, tensor in params.items():
        if tuple(tensor.shape) != (nl, nh) or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Invalid frozen metadata: ' + key)
    key_mode = frozen_factors['key_gain_mode']
    if key_mode not in ('none', 'centered', 'whole'):
        raise ValueError('Unknown frozen key gain mode')
    local_cpu = token_sample(nv, max_tokens, source_identity(row), seed)
    local = local_cpu.to(device)
    units = []
    for index, layer in enumerate(cache.layers):
        key = layer.keys.index_select(2, visual).float()
        value = layer.values.index_select(2, visual)
        mean = key.mean(-2, keepdim=True).half().contiguous()
        z = wht(key - mean.float())
        factor = params.get('key_head_factors')
        factor = None if factor is None else factor[index][None, :, None, None]
        code, gain = encode_key(z, old_book, amplitude='norm', assignment='l2', gain_factor=factor)
        ids = unpack_nibbles(code)
        base = decode_key(code, gain, old_book)
        # Keep original and final mean distinct. The reconstruction target of K
        # residual fitting is z, matching KeyBank, even for whole-mean folding.
        final_mean = (mean.float() * factor).half() if factor is not None and key_mode == 'whole' else mean
        record = binary_centroids(value.transpose(2, 3).contiguous(), group_size=nv)
        if 'value_head_factors' in params:
            record = fold(record, 1, params['value_head_factors'][index][None, :, None, None],
                          params['value_center_offsets'][index][None, :, None, None])
        binary = unpack_binary(record[0], nv).transpose(2, 3).contiguous()
        minimum = record[2].transpose(2, 3).contiguous()
        scale = record[1].transpose(2, 3).contiguous()
        # Use the old deployed decoder for residual generation. A separate
        # Round8 reader identity check must verify the cube arithmetic matches.
        value_base = decode_last_generic(*record, nv, 1).transpose(2, 3).contiguous()
        for head in range(nh):
            item = dict(layer=index, head=head,
                k_source=z[0, head, local].cpu(),
                k_index4=ids[0, head, local].cpu(),
                k_gain=gain[0, head, local].cpu(),
                k_old_base=base[0, head, local].cpu(),
                k_mean=mean[0, head, 0].cpu(),
                k_final_mean=final_mean[0, head, 0].cpu(),
                k_factor=torch.tensor(1., dtype=torch.float32) if factor is None else factor[0, head, 0, 0].cpu(),
                v_source=value[0, head, local].cpu(),
                v_old_base=value_base[0, head, local].cpu(),
                v_binary=binary[0, head, local].cpu(),
                v_min=minimum[0, head, 0].cpu(),
                v_scale=scale[0, head, 0].cpu())
            for axis, source, initial in (('k', z, base), ('v', value, value_base)):
                depth = sum(a == axis and l == index and h == head for a, l, h, _ in stages)
                codes, alphas, _, _ = encode_sign_depth(source[0, head, local], initial[0, head, local], depth)
                item[axis + '_residual_codes'] = [x.cpu() for x in codes]
                item[axis + '_residual_alphas'] = [x.cpu() for x in alphas]
            units.append(item)
        del key, value, z, code, gain, ids, base, binary, value_base, record
    return dict(schema=SCHEMA, source_id=source_identity(row), id=str(row['id']),
                image_id=str(row['image_id']), visual_tokens=nv,
                sampled_token_indices=local_cpu.int(),
                sampled_visual_positions=visual[local].int().cpu(),
                image_boundary=int(visual[-1]) + 1, layers=nl, heads=nh, dimension=128,
                units=units, selected_stages=[list(x) for x in stages],
                diagnostic_source_owner=True, deployable=False,
                frame='FP16 all-image K mean before native WHT; final folded K gain; V all-image frozen fold',
                token_sampling='source-seeded uniform without replacement; frame and old fields precede sampling')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8-sig'))
    a = c['arguments']
    # Refuse an implicit prefix or truncation: a frozen source manifest defines
    # every image. Repeated questions from one source are collected once.
    paths = c.get('sample_paths', [a['samples']])
    rows, seen = [], set()
    for name in paths:
        path = Path(name)
        content = path.read_text(encoding='utf-8-sig')
        parsed = [json.loads(x) for x in content.splitlines() if x.strip()] if path.suffix == '.jsonl' else json.loads(content)
        for row in parsed:
            identity = source_identity(row)
            if identity not in seen:
                rows.append(row)
                seen.add(identity)
    args.output.mkdir(exist_ok=False, parents=True)
    from bank import install_legacy
    install_legacy(a['legacy_scripts'])
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from p180_multiturn_core import attention_bridge
    import session_boundary
    torch.set_num_threads(int(c.get('threads', 8)))
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(a['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    book_path = Path(c['old_book'])
    factor_path = Path(c['frozen_factors'])
    book = torch.tensor(json.loads(book_path.read_text())['centers'], device=model.device, dtype=torch.float16)
    factors = json.loads(factor_path.read_text())
    if factors['encoder_book_sha256'] != sha(book_path):
        raise ValueError('Frozen book/factor identity mismatch')
    entries = []
    with torch.no_grad():
        for number, row in enumerate(rows):
            started = time.perf_counter()
            f = {k: v.to(model.device, dtype=model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in session_boundary.features(processor, row, a.get('max_patches', 16)).items()}
            visual = (f['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
            boundary = int(visual[-1]) + 1
            # Strict image boundary. Slice text-aligned inputs; image tensors
            # remain complete. This equals the native prefix used for visual KV.
            for key in ('input_ids', 'attention_mask', 'token_type_ids', 'position_ids'):
                if key in f:
                    f[key] = f[key][..., :boundary]
            with attention_bridge(model, native=True):
                out = model(**f, use_cache=True, return_dict=True, logits_to_keep=1)
            shard = collect_native_image(out.past_key_values, visual, row, book, factors,
                c['selected_stages'], max_tokens=int(c.get('max_tokens_per_source', 128)), seed=int(c.get('seed', 8)))
            filename = f'source_{number:04d}.pt'
            torch.save(shard, args.output / filename)
            entry = dict(source_id=shard['source_id'], id=shard['id'], image_id=shard['image_id'],
                file=filename, sha256=sha(args.output / filename), visual_tokens=shard['visual_tokens'],
                sampled_tokens=len(shard['sampled_token_indices']), layers=shard['layers'], heads=shard['heads'])
            entries.append(entry)
            print(json.dumps(dict(source=entry['source_id'], sampled_tokens=entry['sampled_tokens'], seconds=time.perf_counter()-started)), flush=True)
            del out, shard, f
    manifest = dict(schema=SCHEMA, entries=entries, sources=len(entries), config_sha256=sha(args.config),
        sample_inputs=[dict(name=Path(x).name, sha256=sha(x)) for x in paths],
        old_book_sha256=sha(book_path), frozen_factors_sha256=sha(factor_path),
        selected_stages=c['selected_stages'], labels_used=False, model_weights_frozen=True,
        diagnostic_source_owner=True, deployable=False, source_script_sha256=sha(__file__),
        scope='bounded native-coordinate calibration sources only; no model quality or serving efficiency claim')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output / 'complete.json').write_text(json.dumps(dict(status='COMPLETE', sources=len(entries))), encoding='utf-8')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
    main()
