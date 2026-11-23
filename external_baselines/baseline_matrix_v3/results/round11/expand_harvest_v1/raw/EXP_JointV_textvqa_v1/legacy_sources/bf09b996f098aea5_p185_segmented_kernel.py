"""P185 FP16 / P186 packed dual-segment attention with original causal positions.

Uses the established tiled online-softmax algorithm (FlashAttention;
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
Our layout reads visual backing and append-only text directly; no KV concatenation.
The packed path reads KIVI2-g128 directly. No split-K or backward support.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read(Q, VK, VV, TK, TV, VP, TP, IDX, O, KS, KM, VS, VM,
          nq, nv, nt, qstart,
          qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr,
          vs0: tl.constexpr, vs1: tl.constexpr,
          ts0: tl.constexpr, ts1: tl.constexpr,
          H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
          PACKED: tl.constexpr, NVFULL: tl.constexpr,
          SELECTED: tl.constexpr, NI: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(1)
    batch = bh // H
    head = bh % H
    kh = head // G
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    cols = tl.arange(0, BN)
    q = tl.load(Q+batch*qs0+head*qs1+rows[:, None]*qs2+dims[None, :], rows[:, None] < nq, 0)
    running_m = tl.full((BM,), -float('inf'), tl.float32)
    running_l = tl.full((BM,), 0., tl.float32)
    acc = tl.full((BM, D), 0., tl.float32)
    visual_blocks = tl.cdiv(nv, BN)
    total_blocks = visual_blocks + tl.cdiv(nt, BN)
    for block in range(total_blocks):
        if block < visual_blocks:
            slot = block * BN + cols
            valid = slot < nv
            if SELECTED:
                offset = tl.load(IDX+(batch*(H//G)+kh)*NI+slot, valid, 0)
            else:
                offset = slot
            pos = tl.load(VP+offset, valid, 0)
            if PACKED:
                owner=batch*(H//G)+kh
                kc=tl.load(VK+owner*D*(NVFULL//16)+dims[:,None]*(NVFULL//16)+offset[None,:]//16,valid[None,:],0)
                ki=(kc >> (2*(offset[None,:]%16))) & 3
                ks=tl.load(KS+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0)
                km=tl.load(KM+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0)
                k=((ki.to(tl.float16)*ks).to(tl.float16)+km).to(tl.float16)
                vc=tl.load(VV+owner*NVFULL*(D//16)+offset[:,None]*(D//16)+dims[None,:]//16,valid[:,None],0)
                vi=(vc >> (2*(dims[None,:]%16))) & 3
                vs=tl.load(VS+owner*NVFULL*(D//128)+offset[:,None]*(D//128)+dims[None,:]//128,valid[:,None],0)
                vm=tl.load(VM+owner*NVFULL*(D//128)+offset[:,None]*(D//128)+dims[None,:]//128,valid[:,None],0)
                v=((vi.to(tl.float16)*vs).to(tl.float16)+vm).to(tl.float16)
            else:
                kp = VK+batch*vs0+kh*vs1+offset[None, :]*D+dims[:, None]
                vp = VV+batch*vs0+kh*vs1+offset[:, None]*D+dims[None, :]
                k = tl.load(kp, valid[None, :], 0)
                v = tl.load(vp, valid[:, None], 0)
        else:
            slot = (block-visual_blocks)*BN+cols
            valid = slot < nt
            pos = tl.load(TP+slot, valid, 0)
            kp = TK+batch*ts0+kh*ts1+slot[None, :]*D+dims[:, None]
            vp = TV+batch*ts0+kh*ts1+slot[:, None]*D+dims[None, :]
            k = tl.load(kp, valid[None, :], 0)
            v = tl.load(vp, valid[:, None], 0)
        score = tl.dot(q, k) * (D ** -.5)
        legal = valid[None, :] & (pos[None, :] <= qstart+rows[:, None])
        score = tl.where(legal, score, -float('inf'))
        new_m = tl.maximum(running_m, tl.max(score, 1))
        safe_m = tl.where(new_m == -float('inf'), 0., new_m)
        alpha = tl.exp(running_m-safe_m)
        weight = tl.exp(score-safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(weight.to(tl.float16), v)
        running_l = running_l*alpha+tl.sum(weight, 1)
        running_m = new_m
    output = acc / tl.maximum(running_l[:, None], 1.e-30)
    tl.store(O+((batch*H+head)*nq+rows[:, None])*D+dims[None, :], output, rows[:, None] < nq)


def segmented_attention(query, state, selected=None, query_start=None):
    if query.dtype != torch.float16 or query.shape[-1] not in (16, 32, 64, 128):
        raise ValueError('FP16 power-of-two head dimensions 16..128 required')
    if query.shape[-1] != state.text_k.shape[-1] or query.stride(-1) != 1:
        raise ValueError('Incompatible query layout')
    b, h, n, d = query.shape
    if h % state.text_k.shape[1]:
        raise ValueError('Invalid GQA ratio')
    groups = h // state.text_k.shape[1]
    packed=hasattr(state,'k_code')
    if packed and (d!=128 or state.visual_pos.numel()%128):
        raise ValueError('Packed g128 path requires head_dim=128 and complete visual groups')
    vk=state.k_code if packed else state.visual_k
    vv=state.v_code if packed else state.visual_v
    meta=[state.k_scale,state.k_min,state.v_scale,state.v_min] if packed else [query]*4
    nv = state.visual_pos.numel() if selected is None else selected.shape[-1]
    output = torch.empty((b, h, n, d), device=query.device, dtype=query.dtype)
    _read[(triton.cdiv(n, 16), b*h)](
        query, vk, vv, state.text_k, state.text_v,
        state.visual_pos, state.text_pos, selected if selected is not None else state.visual_pos, output,
        *meta,
        n, nv, state.text_length, state.length-n if query_start is None else query_start,
        *query.stride()[:3], *vk.stride()[:2], *state.text_k.stride()[:2],
        h, groups, d, packed, state.visual_pos.numel(), selected is not None, nv, 16, 32,
        num_warps=4, num_stages=2, enable_fp_fusion=not packed)
    return output
