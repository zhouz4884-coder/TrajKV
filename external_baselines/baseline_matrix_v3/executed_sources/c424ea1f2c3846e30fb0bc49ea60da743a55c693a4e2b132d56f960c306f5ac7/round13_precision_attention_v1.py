"""All-support mixed K1/K2 reader: two genuine compact index pools.

Every visual row participates in one softmax. V base/residuals and absolute
positions are identical to the request-specific R bank; K has no residuals.
"""
import torch
import triton
import triton.language as tl
from fast_wht import wht


@triton.jit
def _read(Q, QR, BIAS, KL, KH, GAIN, BL, BH, RANK, VC, VS, VM, RC, RA, LV,
          TK, TV, VP, TP, O, nq, nv, nt, qstart,
          QS1: tl.constexpr, QS2: tl.constexpr, TS1: tl.constexpr,
          HN: tl.constexpr, LN: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr):
    head = tl.program_id(1)
    owner = head // G
    index_v = tl.load(LV + owner * 2)
    index_v2 = tl.load(LV + owner * 2 + 1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    q = tl.load(Q + head * QS1 + rows[:, None] * QS2 + dims[None, :], rows[:, None] < nq, 0)
    qr = tl.load(QR + (head * nq + rows[:, None]) * D + dims[None, :], rows[:, None] < nq, 0)
    bias = tl.load(BIAS + head * nq + rows, rows < nq, 0)
    m = tl.full((BM,), -float('inf'), tl.float32)
    total = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    vb = tl.cdiv(nv, BN)
    for block in range(vb + tl.cdiv(nt, BN)):
        if block < vb:
            offset = block * BN + cols
            valid = offset < nv
            rank = tl.load(RANK + offset, valid, -1).to(tl.int32)
            high = rank >= 0
            high_rank = tl.maximum(rank, 0)
            low_rank = tl.maximum(-rank - 1, 0)
            gain = tl.load(GAIN + owner * nv + offset, valid, 0).to(tl.float32)
            low_value = tl.full((D, BN), 0., tl.float32)
            high_value = tl.full((D, BN), 0., tl.float32)
            if LN > 0:
                packed = tl.load(KL + (owner * LN + low_rank[None, :]) * 16 + dims[:, None] // 8,
                                 valid[None, :] & (~high[None, :]), 0).to(tl.int32)
                li = (packed >> (4 * ((dims[:, None] // 4) % 2))) & 15
                low_value = tl.load(BL + li * 4 + dims[:, None] % 4,
                                    valid[None, :] & (~high[None, :]), 0).to(tl.float32)
            if HN > 0:
                hi = tl.load(KH + (owner * HN + high_rank[None, :]) * 32 + dims[:, None] // 4,
                             valid[None, :] & high[None, :], 0).to(tl.int32)
                high_value = tl.load(BH + hi * 4 + dims[:, None] % 4,
                                     valid[None, :] & high[None, :], 0).to(tl.float32)
            k = (tl.where(high[None, :], high_value, low_value) * gain[None, :]).to(tl.float16)
            word_v = tl.load(VC + (owner * D + dims[None, :]) * (nv // 32) + offset[:, None] // 32,
                             valid[:, None], 0)
            code_v = (word_v >> (offset[:, None] % 32)) & 1
            # Metadata covers all 128 valid channels, independent of row tail.
            scale = tl.load(VS + owner * D + dims[None, :])
            minimum = tl.load(VM + owner * D + dims[None, :])
            v = ((code_v.to(tl.float16) * scale).to(tl.float16) + minimum).to(tl.float16)
            if index_v >= 0:
                word_r = tl.load(RC + (index_v * nv + offset[:, None]) * (D // 32) + dims[None, :] // 32,
                                 valid[:, None], 0)
                sign = (word_r >> (dims[None, :] % 32)) & 1
                alpha = tl.load(RA + index_v * nv + offset, valid, 0).to(tl.float32)
                v = (v.to(tl.float32) + (2 * sign.to(tl.float32) - 1) * alpha[:, None]).to(tl.float16)
            if index_v2 >= 0:
                word_r = tl.load(RC + (index_v2 * nv + offset[:, None]) * (D // 32) + dims[None, :] // 32,
                                 valid[:, None], 0)
                sign = (word_r >> (dims[None, :] % 32)) & 1
                alpha = tl.load(RA + index_v2 * nv + offset, valid, 0).to(tl.float32)
                v = (v.to(tl.float32) + (2 * sign.to(tl.float32) - 1) * alpha[:, None]).to(tl.float16)
            score = tl.dot(qr, k) * (D ** -.5) + bias[:, None]
            pos = tl.load(VP + offset, valid, 0)
        else:
            offset = (block - vb) * BN + cols
            valid = offset < nt
            k = tl.load(TK + owner * TS1 + offset[None, :] * D + dims[:, None], valid[None, :], 0)
            v = tl.load(TV + owner * TS1 + offset[:, None] * D + dims[None, :], valid[:, None], 0)
            score = tl.dot(q, k) * (D ** -.5)
            pos = tl.load(TP + offset, valid, 0)
        score = tl.where(valid[None, :] & (pos[None, :] <= qstart + rows[:, None]), score, -float('inf'))
        nm = tl.maximum(m, tl.max(score, 1))
        safe = tl.where(nm == -float('inf'), 0., nm)
        alpha = tl.exp(m - safe)
        weight = tl.exp(score - safe[:, None])
        acc = acc * alpha[:, None] + tl.dot(weight.to(tl.float16), v)
        total = total * alpha + tl.sum(weight, 1)
        m = nm
    tl.store(O + ((head * nq + rows[:, None]) * D + dims[None, :]),
             acc / tl.maximum(total[:, None], 1e-30), rows[:, None] < nq)


def attention(q, state, book, v_group_size=128, visual_bias=None):
    b, h, n, d = q.shape
    nv = state.visual_pos.numel()
    g = h // state.text_k.shape[1]
    nh, nl = int(state.round13_high_count), int(state.round13_low_count)
    if b != 1 or d != 128 or q.dtype != torch.float16 or q.stride(-1) != 1:
        raise ValueError('Round13 token-tier reader requires batch1 FP16 D128')
    if nv != nh + nl or nv % 32 or int(v_group_size) != nv:
        raise ValueError('All original visual rows and full-token V groups required')
    if tuple(book.shape) != (16, 4) or tuple(state.round13_high_book.shape) != (256, 4):
        raise ValueError('Two actual 16x4/256x4 books required')
    if state.direction_code.shape != (1, h // g, nl, 16) or state.round13_high_code.shape != (1, h // g, nh, 32):
        raise ValueError('Two disjoint compact 16/32 byte key pools required')
    if state.direction_code.dtype != torch.uint8 or state.round13_high_code.dtype != torch.uint8:
        raise ValueError('Actual packed uint8 key pools required')
    if any(hasattr(state, name) for name in ('visual_k', 'visual_v')):
        raise ValueError('No native visual source at read')
    if int(getattr(state, 'round13_key_residual_count', 0)):
        raise ValueError('All K residual slots are exchanged, not duplicated')
    qr = wht(q, half=True)
    bias = (q.float() * state.direction_mean.repeat_interleave(g, 1).float()).sum(-1).div(d ** .5)
    if visual_bias is not None:
        bias = bias + visual_bias.repeat_interleave(g)[None, :, None]
    bias = bias.contiguous()
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    rc = getattr(state, 'depth_residual_code', state.v_code)
    ra = getattr(state, 'depth_residual_alpha', state.v_scale)
    _read[(triton.cdiv(n, 16), h)](q, qr, bias, state.direction_code, state.round13_high_code,
        state.direction_gain, book, state.round13_high_book, state.round13_tier_rank,
        state.v_code, state.v_scale, state.v_min, rc, ra, state.depth_residual_lookup[1],
        state.text_k, state.text_v, state.visual_pos, state.text_pos, out,
        n, nv, state.text_length, state.length - n,
        q.stride(1), q.stride(2), state.text_k.stride(1), nh, nl, d, g,
        16, 32, num_warps=4, num_stages=2, enable_fp_fusion=False)
    return out
