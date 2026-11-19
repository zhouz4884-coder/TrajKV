"""Round8 hard codecs: K4/K8 and all-frame V joint8, with true FP16 decode.

CPU import requires only Torch and the frozen sign-depth codec. No source K/V
is retained by these functions. V metadata is supplied after the caller's
unchanged binary-centroid fit and frozen factor fold; no V row gain is added.
"""
import torch

from round6_depth_codec import decode_sign_stage, encode_sign_depth, plan_depth_stages


def key_book_layout(book):
    shape = tuple(book.shape)
    if shape in ((16, 4), (256, 4), (256, 8)):
        words, dim = shape
        return dict(words=words, dimension=dim, groups=128 // dim, positions=False,
                    index_bytes=16 if words == 16 or dim == 8 else 32)
    if shape == (32, 16, 4):
        return dict(words=16, dimension=4, groups=32, positions=True, index_bytes=16)
    raise ValueError('Expected global (16,4)/(256,4)/(256,8) or position-specific (32,16,4) K book')


def pack_nibbles(indices):
    if indices.dtype != torch.uint8 or indices.shape[-1] % 2:
        raise ValueError('Even final uint8 index dimension required')
    if bool((indices > 15).any()):
        raise ValueError('Nibble indices must be below16')
    return (indices[..., 0::2] | (indices[..., 1::2] << 4)).contiguous()


def unpack_nibbles(code):
    if code.dtype != torch.uint8:
        raise ValueError('K index storage must be uint8')
    return torch.stack((code & 15, code >> 4), -1).flatten(-2).contiguous()


def key_template(code, book):
    layout = key_book_layout(book)
    if code.dtype != torch.uint8 or code.shape[-1] != layout['index_bytes']:
        raise ValueError('K payload shape/dtype does not match actual packed layout')
    indices = unpack_nibbles(code) if layout['words'] == 16 else code
    centers = book.half().float()
    if layout['positions']:
        offsets = torch.arange(32, device=code.device) * 16
        chosen = centers.reshape(32 * 16, 4)[indices.long() + offsets]
    else:
        chosen = centers[indices.long()]
    return chosen.flatten(-2)


def decode_key(code, gain, book):
    if gain.dtype != torch.float16:
        raise ValueError('Stored K row gain must be FP16')
    return (key_template(code, book) * gain.float()).half()


def key_amplitude(source, template, mode='norm'):
    source, template = source.float(), template.float()
    if mode == 'norm':
        return source.norm(dim=-1, keepdim=True) / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    dot = (source * template).sum(-1, keepdim=True)
    if mode == 'ls':
        return dot / template.square().sum(-1, keepdim=True).clamp_min(1e-20)
    if mode == 'projection':
        norm = source.norm(dim=-1, keepdim=True)
        fallback = norm / template.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        return torch.where(dot > 1e-12, norm.square() / dot.clamp_min(1e-12), fallback)
    raise ValueError('Round8 first prototype supports norm/ls/projection amplitude')


def _nearest(parts, centers, chunk):
    out = []
    for start in range(0, len(parts), chunk):
        part = parts[start:start + chunk]
        distance = part.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * (part @ centers.T)
        out.append(distance.argmin(-1).byte())
    return torch.cat(out)


def encode_key(source, book, *, amplitude='norm', assignment='l2', gain_factor=None, chunk=65536):
    """Legacy normalized assignment and norm/factor FP16 rounding, generalized.

    source is the actual FP32 centered WHT tensor. The caller owns the FP16
    mean and supplies its source directly; this function never re-centers it.
    """
    if assignment != 'l2':
        raise ValueError('Round8 first prototype uses explicit l2 assignment only')
    if source.shape[-1] != 128 or chunk < 1 or source.numel() == 0:
        raise ValueError('Nonempty D128 source and positive assignment chunk required')
    layout = key_book_layout(book)
    raw, centers = source.float(), book.half().float()
    norm = raw.norm(dim=-1, keepdim=True)
    initial = (norm / 128 ** .5).clamp_min(1e-20)
    normalized = raw / initial
    if layout['positions']:
        parts = normalized.reshape(-1, 32, 4)
        indices = torch.stack([_nearest(parts[:, position], centers[position], chunk)
                               for position in range(32)], -1).reshape(*raw.shape[:-1], 32)
    else:
        parts = normalized.reshape(-1, layout['dimension'])
        indices = _nearest(parts, centers, chunk).reshape(*raw.shape[:-1], layout['groups'])
    code = pack_nibbles(indices) if layout['words'] == 16 else indices.contiguous()
    template = key_template(code, book)
    factor = 1. if gain_factor is None else gain_factor
    # Preserve the two separate stored-gain roundings of the frozen encoder.
    gain = (key_amplitude(raw, template, amplitude).half().float() * factor).half().contiguous()
    return code, gain


def cube_book(device=None):
    """Little-endian channel bits: word=sum(bit[k]*2**k), k=0..7."""
    index = torch.arange(256, device=device, dtype=torch.int64)[:, None]
    shifts = torch.arange(8, device=device, dtype=torch.int64)[None]
    return ((index >> shifts) & 1).half().contiguous()


def cube_indices_from_codes(code, visual_tokens):
    """Transcode actual old bit codes; never re-threshold or nearest-assign."""
    nv = int(visual_tokens)
    if code.dtype != torch.int32 or code.ndim != 4 or code.shape[-2] != 128 or code.shape[-1] != (nv + 31) // 32 or nv <= 0:
        raise ValueError('Expected int32[B,H,128,ceil(real N/32)] actual binary codes')
    positions = torch.arange(nv, device=code.device, dtype=torch.int64)
    word = code.long().index_select(-1, positions // 32)
    bits = ((word >> (positions % 32)) & 1).transpose(-2, -1).contiguous()
    bits = bits.reshape(*bits.shape[:-1], 16, 8)
    shifts = torch.arange(8, device=code.device, dtype=torch.int64)
    return (bits << shifts).sum(-1).byte().contiguous()


def _value_metadata(scale, minimum, prefix):
    expected = (*prefix, 128, 1)
    if tuple(scale.shape) != expected or tuple(minimum.shape) != expected:
        raise ValueError('Round8 joint8 requires unchanged all-frame metadata [B,H,128,1]')
    if scale.dtype != torch.float16 or minimum.dtype != torch.float16:
        raise ValueError('V scale/min must be the actual stored FP16 tensors after fold')
    return scale[..., 0], minimum[..., 0]


def decode_value_joint8(code, scale, minimum, book):
    """Match decode_last_generic: FP16 multiply, then FP16 add; no FMA."""
    if code.dtype != torch.uint8 or code.ndim != 4 or code.shape[-1] != 16 or tuple(book.shape) != (256, 8):
        raise ValueError('V joint8 requires uint8[B,H,real N,16] and one (256,8) book')
    s, m = _value_metadata(scale, minimum, code.shape[:2])
    template = book.half()[code.long()].flatten(-2)
    multiplied = (template.float() * s[:, :, None].float()).half()
    return (multiplied.float() + m[:, :, None].float()).half().contiguous()


def encode_value_joint8(source, scale, minimum, book, *, chunk=2048):
    """Exact word enumeration against native SSE after actual FP16 affine decode.

    Per-channel S^2 weighting is included by decoding each word into native
    coordinates first; this also handles S=0 and FP16 rounding exactly. Codes
    alone own16B/token. The caller keeps its original all/fold metadata.
    """
    if source.dtype != torch.float16 or source.ndim != 4 or source.shape[-1] != 128 or tuple(book.shape) != (256, 8) or chunk < 1:
        raise ValueError('Expected FP16[B,H,N,128], global (256,8) book and positive chunk')
    s, m = _value_metadata(scale, minimum, source.shape[:2])
    n = source.shape[-2]
    blocks = source.reshape(-1, 8).float()
    scales, minima = s.reshape(-1, 8).float(), m.reshape(-1, 8).float()
    centers = book.half().float()
    out = []
    for start in range(0, len(blocks), chunk):
        stop = min(start + chunk, len(blocks))
        index = torch.arange(start, stop, device=source.device)
        metadata_index = (index // (n * 16)) * 16 + index % 16
        ss, mm = scales[metadata_index], minima[metadata_index]
        product = (ss[:, None, :] * centers[None]).half()
        candidate = (product.float() + mm[:, None, :]).half()
        distance = (blocks[start:stop, None, :] - candidate.float()).square().sum(-1)
        out.append(distance.argmin(-1).byte())
    return torch.cat(out).reshape(*source.shape[:-1], 16).contiguous()
