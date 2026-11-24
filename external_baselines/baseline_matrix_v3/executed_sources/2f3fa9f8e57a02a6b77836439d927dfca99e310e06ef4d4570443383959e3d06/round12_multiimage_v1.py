"""Native InternVL3 two-view protocol and auditable image-coordinate mapping.

No independent KV prefixes are concatenated. The native processor constructs
one complete prompt with original image and one magnified crop; every visual
placeholder is included in the returned immutable visual prefix.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch
from PIL import Image

from m1_protocol import M1PreparedBatch, prepare_m1_inputs


def require(value, message):
    if not value:
        raise ValueError('ROUND12_MULTIIMAGE: ' + message)


def _visual_runs(positions):
    values = positions.detach().cpu().tolist()
    runs = []
    for value in values:
        if not runs or value != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(value)
    return runs


def prepare_multiimage_inputs(processor, image, crop, question, *,
                             image_token_id=151667, max_patches=8,
                             device='cuda:0', original_batch=None):
    """Return native two-image batch plus JSON geometry witness.

    Native per-image preprocessing uses cap8 for the original and cap1 for the
    crop. The processor's own placeholder inserter then builds one complete
    two-image input. No pixels are resized through a second, different backend.
    """
    original = original_batch or prepare_m1_inputs(
        processor, image, question, image_token_id=image_token_id,
        max_patches=max_patches, device=device)
    crop_reference = prepare_m1_inputs(processor, crop, question,
        image_token_id=image_token_id, max_patches=1, device=device)
    require(int(crop_reference.visual_positions.numel()) == 256,
            'native cap1 crop must produce exactly 256 visual tokens')
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Original image:\n'},
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': '\nMagnified region of the same image:\n'},
        {'type': 'image', 'image': crop},
        {'type': 'text', 'text': question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    import numpy as np
    op = original.inputs['pixel_values']
    cp = crop_reference.inputs['pixel_values']
    combined_pixels = torch.cat((op, cp), dim=0)
    counts=[int(op.shape[0]),int(cp.shape[0])]
    text, pieces, image_count, video_count = processor._insert_media_placeholders(
        [prompt], combined_pixels, None, counts, [], np.cumsum(counts),
        np.array([0]), np.array([0]))
    require(image_count==2 and video_count==0,'native placeholder formatter image count differs')
    inputs=dict(processor.tokenizer(text,return_tensors='pt',padding_side='left'))
    processor._check_special_mm_tokens(text,inputs,modalities=['image'])
    inputs['pixel_values']=torch.cat(pieces,dim=0)
    inputs={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in inputs.items()}
    vp = (inputs['input_ids'][0] == image_token_id).nonzero(as_tuple=False).flatten()
    runs = _visual_runs(vp)
    nv = int(original.visual_positions.numel())
    require(len(runs) == 2 and list(map(len, runs)) == [nv, 256],
            'actual native image-placeholder runs must be [original_N,256]')
    pixels = inputs.get('pixel_values')
    require(all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (pixels, op, cp)),
            'expected native [tiles,3,448,448] pixel tensors')
    require(tuple(pixels.shape[1:]) == tuple(op.shape[1:]) == tuple(cp.shape[1:]) == (3,448,448)
            and pixels.shape[0] == op.shape[0] + 1 and cp.shape[0] == 1,
            'actual native tile geometry differs')
    require(torch.equal(pixels[:op.shape[0]], op),
            'multiimage changed original-image native pixel tiles')
    require(torch.equal(pixels[-1:], cp),
            'single crop tensor differs from original native cap1 processing')
    boundary = int(vp[-1]) + 1
    positions = torch.arange(inputs['input_ids'].shape[-1], device=device)[None]
    prefix = {**inputs, 'input_ids': inputs['input_ids'][:, :boundary],
              'attention_mask': inputs['attention_mask'][:, :boundary]}
    batch = M1PreparedBatch(inputs, prefix, vp, boundary, positions, prompt)
    witness = dict(schema='trajkv.round12.native_two_view_protocol.v1',
        original_visual_tokens=nv, crop_visual_tokens=256,
        actual_visual_tokens=int(vp.numel()),
        original_tile_count=int(op.shape[0]), crop_tile_count=1,
        visual_group_lengths=[len(x) for x in runs],
        visual_group_first_positions=[x[0] for x in runs],
        original_pixels_bit_equal_to_single_image=True,
        crop_pixels_bit_equal_to_native_cap1=True,
        actual_native_multiimage_full_prefix=True,
        native_per_image_patch_limits=[max_patches,1],
        native_processor_placeholder_inserter_used=True,
        independent_prefix_KV_concatenated=False,
        all_visual_groups_included_in_owner=True,
        new_prefix_may_differ_from_original_due_to_explicit_two_view_template=True,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    return batch, witness


def infer_native_grid(original_size, actual_tiles, *, thumbnail=True, max_patches=8):
    """Infer row-major grid from native tile count and aspect ratio.

    Caller also binds the actual native processor source. A thumbnail is not
    treated as a local tile. Factorization is validated against actual tiles.
    """
    width, height = map(int, original_size)
    require(width > 0 and height > 0 and actual_tiles >= 1, 'invalid image geometry')
    has_thumbnail = bool(thumbnail and actual_tiles > 1)
    blocks = actual_tiles - int(has_thumbnail)
    from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
    cols,rows=get_optimal_tiled_canvas((height,width),(448,448),1,max_patches)
    require(cols * rows + int(has_thumbnail) == actual_tiles,
            'grid factorization differs from native tile count')
    return dict(original_size=[width,height], grid_columns=cols, grid_rows=rows,
        nonthumbnail_tiles=blocks, has_thumbnail=has_thumbnail,
        thumbnail_tile_index=blocks if has_thumbnail else None,
        actual_tiles=actual_tiles, native_tokens_per_tile=256,
        native_canvas_selector='transformers.models.got_ocr2.image_processing_got_ocr2.get_optimal_tiled_canvas',
        max_patches=max_patches,
        flatten_order='row_major_tiles_then_thumbnail; row_major_16x16_per_tile')


def _clip_bbox(center_x, center_y, width, height, *, divisor=4):
    crop_w, crop_h = max(1, round(width/divisor)), max(1, round(height/divisor))
    left = min(max(round(center_x-crop_w/2), 0), width-crop_w)
    top = min(max(round(center_y-crop_h/2), 0), height-crop_h)
    return [int(left),int(top),int(left+crop_w),int(top+crop_h)]


def select_roi(spatial_weights: Sequence[float], original_size, actual_tiles,
               *, random_seed=None, thumbnail=True, max_patches=8):
    """One 1/16-area ROI; random control has identical size/shape.

    Relevant localization maximizes mass in a fixed-size sliding window on the
    original grid at native 16x16 token resolution. The thumbnail is excluded
    from ROI scoring but its probability mass remains separately reported.
    """
    import random
    grid = infer_native_grid(original_size, actual_tiles, thumbnail=thumbnail,max_patches=max_patches)
    require(len(spatial_weights) == actual_tiles*256, 'routing map length differs')
    vals = torch.as_tensor(spatial_weights, dtype=torch.float64)
    require(bool(torch.isfinite(vals).all()) and bool((vals >= 0).all()),
            'nonfinite or negative routing weights')
    width, height = map(int, original_size)
    rows, cols, blocks = grid['grid_rows'], grid['grid_columns'], grid['nonthumbnail_tiles']
    tile_maps = vals[:blocks*256].reshape(rows, cols, 16, 16)
    canvas = tile_maps.permute(0,2,1,3).reshape(rows*16,cols*16)
    kh, kw = max(1, round(canvas.shape[0]/4)), max(1, round(canvas.shape[1]/4))
    if random_seed is not None:
        rng = random.Random(str(random_seed))
        cw,ch=max(1,round(width/4)),max(1,round(height/4))
        left,top=rng.randint(0,width-cw),rng.randint(0,height-ch)
        bbox=[left,top,left+cw,top+ch]
        method='uniform_same_shape_random_seeded'
        selected_mass=None
    else:
        # Integral image evaluates every window without changing tile/token order.
        integ=torch.nn.functional.pad(canvas,(1,0,1,0)).cumsum(0).cumsum(1)
        sums=integ[kh:,kw:]-integ[:-kh,kw:]-integ[kh:,:-kw]+integ[:-kh,:-kw]
        index=int(sums.reshape(-1).argmax())
        top,left=divmod(index,sums.shape[1])
        cx=(left+kw/2)/canvas.shape[1]*width
        cy=(top+kh/2)/canvas.shape[0]*height
        bbox=_clip_bbox(cx,cy,width,height)
        method='max_native_compressed_query_attention_fixed_area'
        selected_mass=float(sums.reshape(-1)[index])
    fraction=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(width*height)
    return dict(schema='trajkv.round12.roi.v1',bbox_xyxy=bbox,
        area_fraction=fraction,method=method,random_seed=str(random_seed) if random_seed is not None else None,
        grid=grid,selected_visual_attention_mass=selected_mass,
        thumbnail_visual_attention_mass=float(vals[blocks*256:].sum()),
        nonthumbnail_visual_attention_mass=float(vals[:blocks*256].sum()),
        approximate_sampling_density_ratio=1/(blocks*fraction),
        crop_is_only_upsampling=(bbox[2]-bbox[0]<448 and bbox[3]-bbox[1]<448),
        question_or_answer_GT_used=False)
