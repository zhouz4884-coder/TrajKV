"""A2 all-support page residual reader; derived from frozen fused_progressive.py.
Mask selects residual payload only, never visual softmax support.
"""
"""P185 FP16 / P186 packed dual-segment attention with original causal positions.

Uses the established tiled online-softmax algorithm (FlashAttention;
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
Our layout reads visual backing and append-only text directly; no KV concatenation.
Local extension: base 1/2/4-bit g128, independent K-r2 / V-sign reads.
1-bit is a TrajKV local extension, NOT official KIVI. No split-K or backward support.
"""
import torch
import triton
import triton.language as tl


def unpack_last_generic(code, bits):
    """Independent uint32-word reference, including local 1-bit codes.

    The official generic Triton encoder stores 32//bits consecutive scalars
    in each int32 word, lowest position in the lowest bits. Zero-extending
    signed words before shifting also exercises a set high/sign bit safely.
    """
    if bits not in (1, 2, 4):
        raise ValueError('Supported local base widths are 1, 2, 4')
    word = code.to(torch.int64) & 0xffffffff
    shifts = torch.arange(32 // bits, device=code.device, dtype=torch.int64) * bits
    values = (word.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return values.reshape(*code.shape[:-1], code.shape[-1] * (32 // bits)).to(torch.int16)


def decode_last_generic(code, scale, minimum, group_size=128, bits=2):
    """Reference matching separate FP16 multiply/add in the fused base path."""
    values = unpack_last_generic(code, bits)
    if values.shape[-1] % group_size:
        raise ValueError('Incomplete reference quantization group')
    groups = values.reshape(*values.shape[:-1], -1, group_size)
    if tuple(scale.shape) != tuple(groups.shape[:-1]) or scale.shape != minimum.shape:
        raise ValueError('Metadata does not match the packed tensor groups')
    decoded = groups.to(scale.dtype) * scale[..., None]
    return (decoded + minimum[..., None]).reshape(values.shape).contiguous()


@triton.jit
def _read(Q, VK, VV, TK, TV, VP, TP, IDX, O, KS, KM, VS, VM, RK, RKS, RKM, RV, RVS, RVM, NORM, MASK,
          nq, nv, nt, qstart,
          qs0: tl.constexpr, qs1: tl.constexpr, qs2: tl.constexpr,
          vs0: tl.constexpr, vs1: tl.constexpr,
          ts0: tl.constexpr, ts1: tl.constexpr,
          H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
          PACKED: tl.constexpr, NVFULL: tl.constexpr, BITS: tl.constexpr, REFINE_K: tl.constexpr, REFINE_V: tl.constexpr, OBSERVE: tl.constexpr,
          SELECTED: tl.constexpr, NI: tl.constexpr, MASKED: tl.constexpr, PAGE: tl.constexpr,VIS_GAIN:tl.constexpr,TXT_GAIN:tl.constexpr,
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
                kc=tl.load(VK+owner*D*(NVFULL//(32//BITS))+dims[:,None]*(NVFULL//(32//BITS))+offset[None,:]//(32//BITS),valid[None,:],0)
                ki=(kc >> (BITS*(offset[None,:]%(32//BITS)))) & ((1<<BITS)-1)
                ks=tl.load(KS+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0)
                km=tl.load(KM+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0)
                k=((ki.to(tl.float16)*ks).to(tl.float16)+km).to(tl.float16)
                vc=tl.load(VV+owner*NVFULL*(D//(32//BITS))+offset[:,None]*(D//(32//BITS))+dims[None,:]//(32//BITS),valid[:,None],0)
                vi=(vc >> (BITS*(dims[None,:]%(32//BITS)))) & ((1<<BITS)-1)
                vs=tl.load(VS+owner*NVFULL*(D//128)+offset[:,None]*(D//128)+dims[None,:]//128,valid[:,None],0)
                vm=tl.load(VM+owner*NVFULL*(D//128)+offset[:,None]*(D//128)+dims[None,:]//128,valid[:,None],0)
                v=((vi.to(tl.float16)*vs).to(tl.float16)+vm).to(tl.float16)
                use_residual = True
                if MASKED:
                    use_residual = tl.load(MASK+owner*tl.cdiv(NVFULL,PAGE)+(block*BN)//PAGE) != 0
                if use_residual:
                    if REFINE_K:
                        total=(H//G)*D*NVFULL//8
                        ix=(owner*D+dims[:,None])*NVFULL+offset[None,:]
                        a=tl.load(RK+ix//8,valid[None,:],0)
                        b=tl.load(RK+total+ix//8,valid[None,:],0)
                        code=((a>>(ix%8))&1)+2*((b>>(ix%8))&1)
                        rs=tl.load(RKS+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0).to(tl.float32)
                        rm=tl.load(RKM+owner*D*(NVFULL//128)+dims[:,None]*(NVFULL//128)+offset[None,:]//128,valid[None,:],0).to(tl.float32)
                        residual=code.to(tl.float32)*rs+rm
                        k=(k.to(tl.float32)+residual).to(tl.float16)
                    if REFINE_V:
                        ix=(owner*NVFULL+offset[:,None])*D+dims[None,:]
                        a=tl.load(RV+ix//8,valid[:,None],0)
                        code=(a>>(ix%8))&1
                        alpha=tl.load(RVS+owner*NVFULL*(D//128)+offset[:,None]*(D//128)+dims[None,:]//128,valid[:,None],0).to(tl.float32)
                        residual=(2*code.to(tl.float32)-1)*alpha
                        v=(v.to(tl.float32)+residual).to(tl.float16)
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
        if VIS_GAIN != 1. or TXT_GAIN != 1.:
            if block<visual_blocks:score=score*VIS_GAIN
            else:score=score*TXT_GAIN
        legal = valid[None, :] & (pos[None, :] <= qstart+rows[:, None])
        score = tl.where(legal, score, -float('inf'))
        new_m = tl.maximum(running_m, tl.max(score, 1))
        safe_m = tl.where(new_m == -float('inf'), 0., new_m)
        alpha = tl.exp(running_m-safe_m)
        weight = tl.exp(score-safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(weight.to(tl.float16), v)
        running_l = running_l*alpha+tl.sum(weight, 1)
        running_m = new_m
    if OBSERVE:
        tl.store(NORM+(batch*H+head)*nq+rows,running_m+tl.log(running_l),rows<nq)
    output = acc / tl.maximum(running_l[:, None], 1.e-30)
    tl.store(O+((batch*H+head)*nq+rows[:, None])*D+dims[None, :], output, rows[:, None] < nq)


def attention(query, state, selected=None, query_start=None, refinement=None, bits=2, observe=False,
              refine_k=True, refine_v=True, refinement_mask=None, page_size=256,visual_logit_gain=1.,text_logit_gain=1.):
    if query.dtype != torch.float16 or query.shape[-1] not in (16, 32, 64, 128):
        raise ValueError('FP16 power-of-two head dimensions 16..128 required')
    if query.shape[-1] != state.text_k.shape[-1] or query.stride(-1) != 1:
        raise ValueError('Incompatible query layout')
    b, h, n, d = query.shape
    if visual_logit_gain<=0 or text_logit_gain<=0:raise ValueError('Positive attention-logit gains required')
    assert b==1 and bits in (1,2,4)
    active_k = refinement is not None and bool(refine_k)
    active_v = refinement is not None and bool(refine_v)
    if (active_k or active_v) and bits not in (1,2):
        raise ValueError("K-r2/V-sign refinement requires a 1/2-bit base")
    if refinement is not None and len(refinement) != 6:
        raise ValueError("Expected six residual tensors: RK,RKS,RKM,RV,RVS,RVM")
    if h % state.text_k.shape[1]:
        raise ValueError('Invalid GQA ratio')
    groups = h // state.text_k.shape[1]
    packed=hasattr(state,'k_code')
    if (active_k or active_v) and not packed:
        raise ValueError('Residual refinement requires a packed visual state')
    if packed and (d!=128 or state.visual_pos.numel()%128):
        raise ValueError('Packed g128 path requires head_dim=128 and complete visual groups')
    if refinement_mask is not None:
        if selected is not None or page_size not in (128,256) or state.visual_pos.numel()%page_size:
            raise ValueError('Masked refinement preserves all support and requires aligned 128/256-token pages')
        expected=(b,state.text_k.shape[1],state.visual_pos.numel()//page_size)
        if tuple(refinement_mask.shape)!=expected or not refinement_mask.is_contiguous():
            raise ValueError('Expected contiguous [batch,kv_head,page] mask')
        if refinement_mask.dtype not in (torch.bool,torch.uint8):raise ValueError('Byte mask required')
    vk=state.k_code if packed else state.visual_k
    vv=state.v_code if packed else state.visual_v
    meta=[state.k_scale,state.k_min,state.v_scale,state.v_min] if packed else [query]*4
    nv = state.visual_pos.numel() if selected is None else selected.shape[-1]
    output = torch.empty((b, h, n, d), device=query.device, dtype=query.dtype)
    norm=torch.empty((b,h,n),device=query.device,dtype=torch.float32) if observe else query
    _read[(triton.cdiv(n, 16), b*h)](
        query, vk, vv, state.text_k, state.text_v,
        state.visual_pos, state.text_pos, selected if selected is not None else state.visual_pos, output,
        *meta, *(refinement if refinement is not None else [query]*6), norm, refinement_mask if refinement_mask is not None else query,
        n, nv, state.text_length, state.length-n if query_start is None else query_start,
        *query.stride()[:3], *vk.stride()[:2], *state.text_k.stride()[:2],
        h, groups, d, packed, state.visual_pos.numel(), bits, active_k, active_v, observe, selected is not None, nv, refinement_mask is not None, page_size,visual_logit_gain,text_logit_gain,16,32,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return (output,norm) if observe else output
