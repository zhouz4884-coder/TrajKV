"""Fixed-final-norm K direction proposals after one existing sign residual.

Only old global FP16 G4_16 and a given-J norm/factor+signMAD tuple are
accepted. The frozen caller owns J's l2 provenance; no argmin is rerun here.
r0 is fixed from that tuple BEFORE any NNLS proposal. No existing
source, reader, book or field is changed by this helper; it returns the same
16B packed J and existing FP16 g/alpha. Sign bytes are read-only.

For fixed J/sign, real NNLS interior/axis proposals are normalized jointly to
r0. Then one32-slot sweep proposes the real full-vector cosine-best word with
current physical g/alpha, INCLUDING alpha*sign cross terms. Each proposal is
jointly normalized in real arithmetic and rounded to stored FP16 g/alpha.
Actual Q16(Q16(g*t)+alpha*s) must strictly improve cosine, not increase supplied
rotated-source SSE and remain within2^-9 of the ORIGINAL r0. Old fields win
ties. This is a single-real-proposal coordinate heuristic, not a global or
hard-FP16 conditional optimum, native-mean-bridge or task guarantee.
"""
import torch

from round8_codec import (key_amplitude, decode_key, key_template,
                         pack_nibbles, unpack_nibbles)
from round6_depth_codec import encode_sign_depth


NORM_REL_TOL = 2. ** -9


def _signs(code):
    shift = torch.arange(32, device=code.device, dtype=torch.int64)
    return (((code.long()[..., None] >> shift) & 1).flatten(-2).float()*2-1)


def _decode(template, gain, alpha, sign):
    base = (template * gain.float()).half()
    return (base.float() + alpha.float()[:, None]*sign).half()


def _metrics(source_double, decoded):
    y = decoded.double()
    norm2 = y.square().sum(-1)
    source2 = source_double.square().sum(-1)
    cosine = (source_double*y).sum(-1)/(source2*norm2).clamp_min(torch.finfo(torch.float64).tiny).sqrt()
    return cosine, (source_double-y).square().sum(-1), norm2.sqrt()


def _real_score(source_double, direction):
    norm2=direction.square().sum(-1)
    return torch.where((norm2>0)&torch.isfinite(norm2),
        (source_double*direction).sum(-1)/norm2.clamp_min(torch.finfo(torch.float64).tiny).sqrt(),
        torch.full_like(norm2,-torch.inf))


def normalized_proposal(source_double, template, sign, real_gain, real_alpha, target_norm,
                        current_cosine, current_sse, eligible):
    """CPU-testable actual physical guard; r0 is never updated here."""
    direction=real_gain[:,None]*template.double()+real_alpha[:,None]*sign.double()
    norm=direction.square().sum(-1).sqrt()
    valid=eligible&torch.isfinite(real_gain)&torch.isfinite(real_alpha)&(real_gain>=0)&(real_alpha>=0)&torch.isfinite(norm)&(norm>0)
    scale=torch.where(valid,target_norm/norm.clamp_min(torch.finfo(torch.float64).tiny),torch.zeros_like(norm))
    gain=(real_gain*scale).half()[:,None]
    alpha=(real_alpha*scale).half()
    decoded=_decode(template,gain,alpha,sign)
    cosine,sse,actual_norm=_metrics(source_double,decoded)
    relative=(actual_norm/target_norm.clamp_min(torch.finfo(torch.float64).tiny)-1).abs()
    finite=torch.isfinite(gain[:,0])&torch.isfinite(alpha)&torch.isfinite(decoded).all(-1)&torch.isfinite(cosine)&torch.isfinite(sse)
    cos_ok=cosine>current_cosine
    norm_ok=relative<=NORM_REL_TOL
    sse_ok=sse<=current_sse
    accept=valid&finite&cos_ok&norm_ok&sse_ok
    checks=dict(proposals=int(eligible.sum()),accepted=int(accept.sum()),
        invalid_direction=int((eligible&(~valid)).sum()),
        failed_nonfinite=int((valid&(~finite)).sum()),
        failed_cosine=int((valid&finite&(~cos_ok)).sum()),
        failed_fixed_norm=int((valid&finite&(~norm_ok)).sum()),
        failed_source_SSE=int((valid&finite&(~sse_ok)).sum()))
    return gain,alpha,cosine,sse,actual_norm,accept,checks


def real_nnls_candidates(source_double, template, sign):
    """Real two-column NNLS directions, not old hard-refit outputs."""
    t,s=template.double(),sign.double()
    T,B,D=t.square().sum(-1),(t*s).sum(-1),s.square().sum(-1)
    u,v=(source_double*t).sum(-1),(source_double*s).sum(-1)
    determinant=T*D-B.square()
    safe=torch.where(determinant>0,determinant,torch.ones_like(determinant))
    g,a=(D*u-B*v)/safe,(T*v-B*u)/safe
    inside=(determinant>0)&(g>=0)&(a>=0)
    zero=torch.zeros_like(T)
    return [('interior',g,a,inside),
        ('gain_axis',(u/T.clamp_min(torch.finfo(torch.float64).tiny)).clamp_min(0),zero,T>0),
        ('alpha_axis',zero,(v/D.clamp_min(torch.finfo(torch.float64).tiny)).clamp_min(0),D>0)]


def slot_real_scores(source_double, template, sign, gain, alpha, book, slot):
    """Incremental full(real g*t+alpha*s) moments, including residual cross term."""
    gd,ad=gain.double()[:,0],alpha.double()
    direction=gd[:,None]*template.double()+ad[:,None]*sign.double()
    A=(source_double*direction).sum(-1)
    U=direction.square().sum(-1)
    sl=slice(4*slot,4*slot+4)
    delta=book.double()[None]-template[:,None,sl].double()
    candidate_A=A[:,None]+gd[:,None]*(source_double[:,None,sl]*delta).sum(-1)
    candidate_U=U[:,None]+2*gd[:,None]*(direction[:,None,sl]*delta).sum(-1)+gd[:,None].square()*delta.square().sum(-1)
    return torch.where((candidate_U>0)&torch.isfinite(candidate_U),
        candidate_A/candidate_U.clamp_min(torch.finfo(torch.float64).tiny).sqrt(),
        torch.full_like(candidate_A,-torch.inf)),_real_score(source_double,direction)


@torch.no_grad()
def recode_key_postres_direction(source, book, initial_code, initial_gain, sign_code,
                                 initial_alpha, *, gain_factor, row_chunk=256):
    """Return packedJ[N,16], g[N,1], alpha[N], scalar statistics.

    source is actual FP32 centered-WHT[N,128], book physical FP16[16,4]. Pass
    the actual frozen factor explicitly (None declares no factor). The helper
    rebuilds norm/factor plus signMAD from the GIVEN J and rejects any initial
    physical-field mismatch, including an ordinary NNLS result. It does not
    rerun argmin with a different batch/chunk shape. The frozen l2 bank owns
    initial J provenance. A later bank subclass
    must keep joint_gain_refit=False and invoke only selected first K heads.
    No bank/reader integration or model execution is performed in this module.
    """
    if source.ndim!=2 or source.shape[-1]!=128 or source.dtype!=torch.float32 or tuple(book.shape)!=(16,4) or book.dtype!=torch.float16:
        raise ValueError('Actual FP32[N,128] centered-WHT source and old FP16[16,4] book required')
    n=len(source)
    if n<1 or row_chunk<1:
        raise ValueError('Positive rows/chunks required')
    if initial_code.shape!=(n,16) or initial_code.dtype!=torch.uint8 or initial_gain.shape!=(n,1) or initial_gain.dtype!=torch.float16 or sign_code.shape!=(n,4) or sign_code.dtype!=torch.int32 or initial_alpha.shape!=(n,) or initial_alpha.dtype!=torch.float16:
        raise ValueError('Same existing uint8 J16B, FP16 gain/alpha and int32 sign fields required')
    if any(t.device!=source.device for t in (book,initial_code,initial_gain,sign_code,initial_alpha)) or any(not bool(torch.isfinite(t).all()) for t in (source,book,initial_gain,initial_alpha)):
        raise ValueError('Finite input fields on one device required')
    if gain_factor is not None:
        factor=torch.as_tensor(gain_factor,device=source.device,dtype=torch.float32)
        if not bool(torch.isfinite(factor).all()) or bool((factor<=0).any()):
            raise ValueError('Actual finite positive frozen factor required')
    original_template=key_template(initial_code,book)
    factor=1. if gain_factor is None else gain_factor
    expected_gain=(key_amplitude(source,original_template,'norm').half().float()*factor).half()
    if not torch.equal(initial_gain,expected_gain):
        raise ValueError('Initial gain is not given-J norm/FP16/factor/FP16; no prior joint NNLS')
    original_base=decode_key(initial_code,initial_gain,book)
    rc,ra,_,_=encode_sign_depth(source,original_base,1)
    if not torch.equal(sign_code,rc[0]) or not torch.equal(initial_alpha,ra[0]):
        raise ValueError('Initial sign/alpha is not original actual-base signMAD; no prior joint NNLS')
    sign=_signs(sign_code)
    original_y=_decode(original_template,initial_gain,initial_alpha,sign)
    if not bool(torch.isfinite(original_y).all()):
        raise ValueError('Finite original final decode required for a fixed norm target')
    sd=source.double()
    original_cos,original_error,target_norm=_metrics(sd,original_y)
    source_norm=sd.square().sum(-1).sqrt()
    valid=(target_norm>0)&(source_norm>0)
    indices=unpack_nibbles(initial_code).clone()
    output_gain,output_alpha=initial_gain.clone(),initial_alpha.clone()
    final_cos,final_error,final_norm=original_cos.clone(),original_error.clone(),target_norm.clone()
    counters={name:0 for name in ('proposals','accepted','invalid_direction','failed_nonfinite','failed_cosine','failed_fixed_norm','failed_source_SSE')}
    by_kind={name:0 for name in ('interior','gain_axis','alpha_axis','slot')}
    strict_real_slot_proposals=0
    for start in range(0,n,row_chunk):
        stop=min(start+row_chunk,n)
        z,s,t=sd[start:stop],sign[start:stop],original_template[start:stop].clone()
        g,a=output_gain[start:stop].clone(),output_alpha[start:stop].clone()
        ids=indices[start:stop].clone()
        r0,active=target_norm[start:stop],valid[start:stop]
        cos,error,norm=final_cos[start:stop],final_error[start:stop],final_norm[start:stop]
        def evaluate(candidate_t,real_g,real_a,eligible,kind):
            nonlocal t,g,a,cos,error,norm
            ng,na,nc,ne,nn,accept,counts=normalized_proposal(z,candidate_t,s,real_g,real_a,r0,cos,error,eligible&active)
            for key,value in counts.items():counters[key]+=value
            by_kind[kind]+=int(accept.sum())
            t=torch.where(accept[:,None],candidate_t,t)
            g=torch.where(accept[:,None],ng,g);a=torch.where(accept,na,a)
            cos=torch.where(accept,nc,cos);error=torch.where(accept,ne,error);norm=torch.where(accept,nn,norm)
            return accept
        for name,real_g,real_a,feasible in real_nnls_candidates(z,t,s):
            evaluate(t,real_g,real_a,feasible,name)
        local_rows=torch.arange(stop-start,device=source.device)
        for slot in range(32):
            scores,current_real_score=slot_real_scores(z,t,s,g,a,book,slot)
            scores[local_rows,ids[:,slot].long()]=current_real_score
            scores=torch.where(active[:,None],scores,torch.full_like(scores,-torch.inf))
            best_score,best=scores.max(-1)
            proposal=active&(best!=ids[:,slot].long())&(best_score>current_real_score)
            candidate_t=t.clone();sl=slice(slot*4,slot*4+4)
            candidate_t[:,sl]=torch.where(proposal[:,None],book.float()[best],t[:,sl])
            direction=g.double()*candidate_t.double()+a.double()[:,None]*s.double()
            proposal&=_real_score(z,direction)>current_real_score
            strict_real_slot_proposals+=int(proposal.sum())
            accept=evaluate(candidate_t,g.double()[:,0],a.double(),proposal,'slot')
            ids[:,slot]=torch.where(accept,best.byte(),ids[:,slot])
        indices[start:stop]=ids;output_gain[start:stop]=g;output_alpha[start:stop]=a
        final_cos[start:stop]=cos;final_error[start:stop]=error;final_norm[start:stop]=norm
    output_code=pack_nibbles(indices)
    final_base=decode_key(output_code,output_gain,book)
    relative=torch.where(valid,(final_norm/target_norm.clamp_min(torch.finfo(torch.float64).tiny)-1).abs(),0.)
    changed=(output_code!=initial_code).any(-1)|(output_gain!=initial_gain).any(-1)|(output_alpha!=initial_alpha)
    stats=dict(schema='trajkv.round8.postres-fixed-norm-direction.v1',rows=n,active_rows=int(valid.sum()),
        zero_source_or_target_noop_rows=int((~valid).sum()),changed_rows=int(changed.sum()),
        initialization_verified_bitwise=True,initial_given_J_not_reassigned=True,
        initial_J_provenance='frozen caller l2 bank/config; this helper does not rerun argmin',
        initialization='given-J norm->FP16->actual frozen factor->FP16, then actual-base signMAD verified bitwise; no prior joint NNLS',
        target='original actual final centered-WHT reconstruction norm fixed before all proposals',
        norm_relative_tolerance=NORM_REL_TOL,max_relative_final_norm_error=float(relative.max()),
        source_sse_before=float(original_error.sum()),source_sse_after=float(final_error.sum()),
        source_base_sse_before=float((sd-original_base.double()).square().sum()),
        source_base_sse_after=float((sd-final_base.double()).square().sum()),
        max_source_row_SSE_increase=max(0.,float((final_error-original_error).max())),
        actual_cosine_before_mean=float(original_cos[valid].mean()) if bool(valid.any()) else None,
        actual_cosine_after_mean=float(final_cos[valid].mean()) if bool(valid.any()) else None,
        min_actual_cosine_change=float((final_cos[valid]-original_cos[valid]).min()) if bool(valid.any()) else None,
        norm_before_mean=float(target_norm.mean()),norm_after_mean=float(final_norm.mean()),
        strict_real_slot_proposals=strict_real_slot_proposals,candidate_guard_counts=counters,accepted_by_kind=by_kind,
        code_bytes_per_token=16,optimized_field_bytes_before=initial_code.numel()+2*(initial_gain.numel()+initial_alpha.numel()),
        optimized_field_bytes_after=output_code.numel()+2*(output_gain.numel()+output_alpha.numel()),
        added_retained_tensor_bytes=0,sign_recode=False,input_fields_mutated=False,
        decode='Q16(Q16(stored_effective_gain*template_J)+stored_alpha*fixed_sign)',
        acceptance='actual final cosine strictly increases, original-r0 norm error<=2^-9, supplied-source row SSE does not increase; original tuple wins ties',
        proposal_scope='real NNLS normalized directions then one32-slot real-full-cosine-best proposal each; not a hard-FP16 conditional/global optimum',
        metric_scope='supplied centered-WHT coordinate SSE; not original native K after mean/bridge, not attention/task quality',
        compute_scope='extra encoder recomputation/FP64 reductions/temporary scratch paid; same reader and persistent field shapes')
    return output_code,output_gain.contiguous(),output_alpha.contiguous(),stats
