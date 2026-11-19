"""Complete residual-state common scaling in existing K gain/alpha slots.

J and sign never change. c is computed from the ORIGINAL actual final y0,
then both existing physical g and alpha receive c before FP16 rounding.
Every mode keeps old fields first and accepts only its own actual objective:
mse=SSE; norm=absolute norm mismatch; projection=absolute source-self-dot
mismatch. Norm/projection deliberately have NO additional SSE gate.

The ideal ray is c*y0; nested FP16 field decoding only approximately follows
it, so actual objectives and direction drift are measured. Projection refers
to this supplied source z, never arbitrary queries or an unbiased estimator.
"""
import torch

from round8_codec import key_template,key_amplitude,decode_key
from round6_depth_codec import encode_sign_depth


MODES=('off','mse','norm','projection')


def signs_from_code(code):
    shifts=torch.arange(32,device=code.device,dtype=torch.int64)
    return (((code.long()[...,None]>>shifts)&1).flatten(-2).float()*2-1)


def final_decode(template,gain,alpha,sign):
    base=(template*gain.float()).half()
    return (base.float()+alpha.float()[:,None]*sign).half()


def metrics(source_double,decoded):
    y=decoded.double();z2=source_double.square().sum(-1);y2=y.square().sum(-1)
    dot=(source_double*y).sum(-1)
    return dict(sse=(source_double-y).square().sum(-1),norm=(y2.sqrt()-z2.sqrt()).abs(),
        projection=(dot-z2).abs(),dot=dot,z2=z2,y2=y2)


@torch.no_grad()
def physical_ray_scale(source,template,gain,sign,alpha,mode):
    """Generic CPU-testable physical ray proposal; no initializer provenance."""
    if mode not in MODES:raise ValueError('Explicit off/mse/norm/projection mode required')
    z=source.double();old_y=final_decode(template,gain,alpha,sign);old=metrics(z,old_y)
    valid=(torch.isfinite(old_y).all(-1)&torch.isfinite(old['z2'])&torch.isfinite(old['y2'])
        &torch.isfinite(old['dot'])&(old['z2']>0)&(old['y2']>0)&(old['dot']>0))
    tiny=torch.finfo(torch.float64).tiny
    if mode=='mse':c=old['dot']/old['y2'].clamp_min(tiny)
    elif mode=='norm':c=(old['z2']/old['y2'].clamp_min(tiny)).sqrt()
    elif mode=='projection':c=old['z2']/old['dot'].clamp_min(tiny)
    else:c=torch.ones_like(old['dot'])
    valid&=torch.isfinite(c)&(c>0)
    safe_c=torch.where(valid,c,torch.ones_like(c))
    candidate_gain=(gain.double()*safe_c[:,None]).half()
    candidate_alpha=(alpha.double()*safe_c).half()
    candidate_y=final_decode(template,candidate_gain,candidate_alpha,sign)
    new=metrics(z,candidate_y)
    physical=(torch.isfinite(candidate_gain[:,0])&torch.isfinite(candidate_alpha)&torch.isfinite(candidate_y).all(-1)
              &torch.isfinite(new['y2'])&(new['y2']>0)&torch.isfinite(new['sse'])
              &torch.isfinite(new['norm'])&torch.isfinite(new['projection']))
    # No implicit MSE filter in the other two objectives.
    objective='sse' if mode=='mse' else mode
    accept=(valid&physical&(new[objective]<old[objective])) if mode!='off' else torch.zeros_like(valid)
    result_gain=torch.where(accept[:,None],candidate_gain,gain)
    result_alpha=torch.where(accept,candidate_alpha,alpha)
    final_y=final_decode(template,result_gain,result_alpha,sign);after=metrics(z,final_y)
    oldd,finald=old_y.double(),final_y.double()
    direction_valid=(old['y2']>0)&(after['y2']>0)&torch.isfinite(oldd).all(-1)&torch.isfinite(finald).all(-1)
    direction_cos=(oldd*finald).sum(-1)/(old['y2']*after['y2']).clamp_min(tiny).sqrt()
    def finite_sum(value):return float(value.sum()) if bool(torch.isfinite(value).all()) else None
    def mean_valid(value,mask):return float(value[mask].mean()) if bool(mask.any()) else None
    stats=dict(schema='trajkv.round8.final-state-ray.v1',mode=mode,rows=len(source),
        accepted_rows=int(accept.sum()),fallback_input_zero_nonpositive_dot_rows=int((~valid).sum()),
        fallback_nonfinite_or_zero_candidate_rows=int((valid&(~physical)).sum()),
        old_tuple_wins_ties=True,extra_SSE_gate=False,SSE_is_acceptance_objective=mode=='mse',
        scale_formula={'off':'1 (original tuple)','mse':'dot(z,y0)/norm(y0)^2',
            'norm':'norm(z)/norm(y0)','projection':'norm(z)^2/dot(z,y0)'}[mode],
        scale_mean=mean_valid(c,valid),scale_min=float(c[valid].min()) if bool(valid.any()) else None,
        scale_max=float(c[valid].max()) if bool(valid.any()) else None,
        sse_before=finite_sum(old['sse']),sse_after=finite_sum(after['sse']),
        norm_error_before=finite_sum(old['norm']),norm_error_after=finite_sum(after['norm']),
        source_projection_error_before=finite_sum(old['projection']),source_projection_error_after=finite_sum(after['projection']),
        norm_before_mean=mean_valid(old['y2'].sqrt(),torch.isfinite(old['y2'])),
        norm_after_mean=mean_valid(after['y2'].sqrt(),torch.isfinite(after['y2'])),
        accepted_rows_with_SSE_increase=int((accept&(after['sse']>old['sse'])).sum()),
        old_to_final_direction_cosine_mean=mean_valid(direction_cos,direction_valid),
        max_old_to_final_direction_cosine_loss=max(0.,float((1-direction_cos[direction_valid]).max())) if bool(direction_valid.any()) else None,
        physical_optimized_bytes_before=2*(gain.numel()+alpha.numel()),
        physical_optimized_bytes_after=2*(result_gain.numel()+result_alpha.numel()),
        added_retained_tensor_bytes=0,indices_changed=False,sign_recode=False,
        coefficient_policy='one common c before independent FP16 field rounding; no second factor fold',
        objective_scope='actual nested-half objective selected by mode; norm/projection may increase SSE',
        projection_scope='self-source projection only; no arbitrary-query/unbiased guarantee',
        direction_scope='ideal real ray; actual FP16 coefficient/base/residual rounding can change direction, measured above',
        metric_scope='supplied centered-WHT source, not mean/native bridge or task quality')
    return result_gain.contiguous(),result_alpha.contiguous(),stats


@torch.no_grad()
def refit_key_final_ray(source,book,code,gain,sign_code,alpha,*,mode,gain_factor):
    """Verified given-J norm/fold/signMAD input, same FP16 g/alpha output.

    source FP32[N,128], old book FP16[16,4], J uint8[N,16], gain FP16[N,1],
    packed sign int32[N,4], alpha FP16[N]. Caller owns original l2 J provenance.
    Explicit actual factor or None verifies initialization without another
    argmin. Nonpositive-dot/zero/nonfinite/overflowing candidates preserve old.
    """
    if source.ndim!=2 or source.shape[-1]!=128 or source.dtype!=torch.float32 or tuple(book.shape)!=(16,4) or book.dtype!=torch.float16:
        raise ValueError('FP32[N,128] source and old FP16[16,4] book required')
    n=len(source)
    if n<1 or code.shape!=(n,16) or code.dtype!=torch.uint8 or gain.shape!=(n,1) or gain.dtype!=torch.float16 or sign_code.shape!=(n,4) or sign_code.dtype!=torch.int32 or alpha.shape!=(n,) or alpha.dtype!=torch.float16:
        raise ValueError('Actual existing packed J/g/sign/alpha fields required')
    if any(t.device!=source.device for t in (book,code,gain,sign_code,alpha)) or any(not bool(torch.isfinite(t).all()) for t in (source,book,gain,alpha)):
        raise ValueError('Finite physical inputs on one device required')
    template=key_template(code,book)
    if gain_factor is None:factor=1.
    else:
        factor=torch.as_tensor(gain_factor,device=source.device,dtype=torch.float32)
        if not bool(torch.isfinite(factor).all()) or bool((factor<=0).any()):raise ValueError('Actual positive frozen factor required')
    expected=(key_amplitude(source,template,'norm').half().float()*factor).half()
    if not torch.equal(gain,expected):raise ValueError('Input gain is not original given-J norm/fold; no preceding NNLS')
    rc,ra,_,_=encode_sign_depth(source,decode_key(code,gain,book),1)
    if not torch.equal(sign_code,rc[0]) or not torch.equal(alpha,ra[0]):raise ValueError('Input is not original actual-base signMAD')
    result_gain,result_alpha,stats=physical_ray_scale(source,template,gain,signs_from_code(sign_code),alpha,mode)
    stats.update(initialization_verified_bitwise=True,initial_given_J_not_reassigned=True,
        initialization='given J original norm->FP16->actual frozen factor->FP16 and original actual-base signMAD',
        initialization_J_provenance='frozen l2 caller; no re-argmin in this helper')
    # Available for the caller to update the existing selected K distortion.
    base=decode_key(code,result_gain,book)
    stats['source_base_sse_after']=float((source.double()-base.double()).square().sum())
    stats['source_sse_after']=stats['sse_after']
    return result_gain,result_alpha,stats
