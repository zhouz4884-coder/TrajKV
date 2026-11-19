"""CPU dense-real conditional oracle and physical guards; no model/GPU/fit."""
import argparse
import json
import math
from pathlib import Path

import torch

from round8_codec import encode_key,decode_key,key_template,pack_nibbles,unpack_nibbles
from round6_depth_codec import encode_sign_depth,decode_sign_stage
from round8_postres_direction import (recode_key_postres_direction,slot_real_scores,
    normalized_proposal,NORM_REL_TOL)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial(source,book,factor):
    code,gain=encode_key(source,book,gain_factor=factor)
    signs,alphas,_,_=encode_sign_depth(source,decode_key(code,gain,book),1)
    return code,gain,signs[0],alphas[0]


def signs_from(code):
    shift=torch.arange(32,dtype=torch.int64)
    return (((code.long()[...,None]>>shift)&1).flatten(-2).float()*2-1)


def dense_one_row(source,book,factor):
    """Independent full128 vector enumeration; same single-real-best policy.

    Separately enumerate the physical feasible changed-word family to measure
    a possible gap. The implementation is NOT required to equal that stronger
    hard oracle; only its real-proposal protocol is the matched oracle.
    """
    code,gain,sign_code,alpha=initial(source,book,factor)
    ids=unpack_nibbles(code).clone();t=key_template(code,book);s=signs_from(sign_code)
    z=source.double()
    def decoded(template,g,a):
        base=(template*g.float()).half()
        return (base.float()+a.float()[:,None]*s).half()
    def metrics(y):
        yd=y.double();norm=yd.square().sum().sqrt()
        cosine=(z*yd).sum()/(z.square().sum().sqrt()*norm)
        return float(cosine),float((z-yd).square().sum()),float(norm)
    current=metrics(decoded(t,gain,alpha));r0=current[2]
    def proposal(template,g,a):
        direction=g*template.double()+a*s.double()
        norm=float(direction.square().sum().sqrt())
        if norm<=0 or not math.isfinite(norm) or g<0 or a<0:
            return None
        ng=torch.tensor([[g*r0/norm]],dtype=torch.float64).half()
        na=torch.tensor([a*r0/norm],dtype=torch.float64).half()
        nc=metrics(decoded(template,ng,na))
        if all(math.isfinite(v) for v in nc) and nc[0]>current[0] and nc[1]<=current[1] and abs(nc[2]/r0-1)<=NORM_REL_TOL:
            return ng,na,nc
        return None
    td,sd=t.double(),s.double()
    T,B,D=float(td.square().sum()),float((td*sd).sum()),float(sd.square().sum())
    u,v=float((z*td).sum()),float((z*sd).sum())
    determinant=T*D-B*B
    candidates=[]
    if determinant>0:
        g,a=(D*u-B*v)/determinant,(T*v-B*u)/determinant
        if g>=0 and a>=0:candidates.append((g,a))
    if T>0:candidates.append((max(u/T,0),0.))
    if D>0:candidates.append((0.,max(v/D,0)))
    for g,a in candidates:
        result=proposal(t,g,a)
        if result is not None:gain,alpha,current=result
    hard_gaps=0;maximum_gap=0.
    for slot in range(32):
        sl=slice(slot*4,slot*4+4);old_id=int(ids[0,slot])
        g,a=float(gain),float(alpha)
        def real_score(template):
            direction=g*template.double()+a*s.double()
            return float((z*direction).sum()/direction.square().sum().sqrt())
        best_id,best_score,best_t=old_id,real_score(t),t
        hard_best_cos=current[0]
        for word in range(16):
            if word==old_id:continue
            candidate=t.clone();candidate[:,sl]=book[word].float()
            score=real_score(candidate)
            if score>best_score:best_id,best_score,best_t=word,score,candidate
            hard=proposal(candidate,g,a)
            if hard is not None:hard_best_cos=max(hard_best_cos,hard[2][0])
        result=proposal(best_t,g,a) if best_id!=old_id else None
        selected_cos=result[2][0] if result is not None else current[0]
        if hard_best_cos>selected_cos:
            hard_gaps+=1;maximum_gap=max(maximum_gap,hard_best_cos-selected_cos)
        if result is not None:
            gain,alpha,current=result;t=best_t;ids[0,slot]=best_id
    return pack_nibbles(ids),gain,alpha,dict(hard_oracle_gap_slots=hard_gaps,max_hard_oracle_cosine_gap=maximum_gap,
        hard_oracle_scope='old tuple plus15 changed-word normalized physical proposals, separately measured; not the implementation oracle')


def rejected(call):
    try:call()
    except (ValueError,RuntimeError):return True
    return False


def controls():
    torch.set_num_threads(1);torch.manual_seed(20261118)
    dense=[]
    for case in range(6):
        source=torch.randn(1,128)*(1+case*.25);book=torch.randn(16,4).half();factor=.93+case*.047
        fields=initial(source,book,factor)
        copies=[value.clone() for value in (source,book,*fields)]
        code,gain,alpha,stats=recode_key_postres_direction(source,book,*fields,gain_factor=factor,row_chunk=1)
        ec,eg,ea,oracle=dense_one_row(source,book,factor)
        assert torch.equal(code,ec) and torch.equal(gain,eg) and torch.equal(alpha,ea)
        assert stats['max_relative_final_norm_error']<=NORM_REL_TOL and stats['max_source_row_SSE_increase']==0
        assert all(torch.equal(a,b) for a,b in zip(copies,(source,book,*fields)))
        assert code.numel()==16 and gain.dtype==torch.float16 and alpha.dtype==torch.float16
        dense.append(dict(case=case,stats=stats,**oracle))
    # Independently enumerate all16 full real directions for each conditioned
    # slot. Incremental moments must include the alpha*sign cross term.
    source=torch.randn(5,128);book=torch.randn(16,4).half();fields=initial(source,book,1.073)
    t=key_template(fields[0],book);s=signs_from(fields[2]);g,a=fields[1],fields[3]
    moments=0
    for slot in (0,7,31):
        scores,_=slot_real_scores(source.double(),t,s,g,a,book,slot)
        all_scores=[]
        for word in range(16):
            candidate=t.clone();candidate[:,slot*4:slot*4+4]=book[word].float()
            y=g.double()*candidate.double()+a.double()[:,None]*s.double()
            all_scores.append((source.double()*y).sum(-1)/y.square().sum(-1).sqrt())
        expected=torch.stack(all_scores,-1)
        assert torch.allclose(scores,expected,rtol=1e-12,atol=1e-12)
        moments+=scores.numel()
    # Zero target and exact physical ties preserve every original bit.
    for source,book in ((torch.zeros(3,128),torch.randn(16,4).half()),
                        (torch.full((2,128),2.),torch.ones(16,4).half())):
        fields=initial(source,book,None)
        code,gain,alpha,stats=recode_key_postres_direction(source,book,*fields,gain_factor=None)
        assert torch.equal(code,fields[0]) and torch.equal(gain,fields[1]) and torch.equal(alpha,fields[3])
    # Wrong factor or a prior changed gain/alpha/sign cannot redefine the norm anchor.
    source=torch.randn(2,128);book=torch.randn(16,4).half();fields=initial(source,book,1.09)
    wrong_gain=(fields[1].float()*.9).half()
    assert rejected(lambda:recode_key_postres_direction(source,book,fields[0],wrong_gain,fields[2],fields[3],gain_factor=1.09))
    wrong_alpha=(fields[3].float()+.125).half()
    assert rejected(lambda:recode_key_postres_direction(source,book,fields[0],fields[1],fields[2],wrong_alpha,gain_factor=1.09))
    assert rejected(lambda:recode_key_postres_direction(source,book,*fields,gain_factor=None))
    # Near-canceling coefficient directions can be norm-infeasible or overflow
    # after stored-field rounding. Such candidates cannot displace the old tuple.
    z=torch.ones(1,128,dtype=torch.float64);t=-torch.ones(1,128);s=torch.ones_like(t)
    target=torch.tensor([128**.5],dtype=torch.float64);old_cos=torch.tensor([.9],dtype=torch.float64);old_error=torch.tensor([100.],dtype=torch.float64)
    _,_,_,_,_,accept,checks=normalized_proposal(z,t,s,torch.tensor([65504.],dtype=torch.float64),
        torch.tensor([65505.],dtype=torch.float64),target,old_cos,old_error,torch.ones(1,dtype=torch.bool))
    assert not bool(accept.any()) and checks['failed_fixed_norm']==1
    _,_,_,_,_,accept,checks_overflow=normalized_proposal(z,t,s,torch.tensor([1e9],dtype=torch.float64),
        torch.tensor([1e9+1],dtype=torch.float64),target,old_cos,old_error,torch.ones(1,dtype=torch.bool))
    assert not bool(accept.any()) and checks_overflow['failed_nonfinite']==1
    # A geometric physical-field guard witness: norm tolerance and cosine
    # alone do not imply SSE descent. This tests the guard, not initialization.
    old_y=torch.ones(1,128);candidate_t=old_y.clone();candidate_t[0,0]=1+2**-10
    z=.5*candidate_t.double();target=old_y.double().square().sum(-1).sqrt()
    old_cos=(z*old_y.double()).sum(-1)/(z.square().sum(-1)*old_y.double().square().sum(-1)).sqrt()
    old_error=(z-old_y.double()).square().sum(-1)
    _,_,new_cos,new_sse,new_norm,accept,checks_sse=normalized_proposal(z,candidate_t,torch.ones_like(candidate_t),
        torch.ones(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),target,old_cos,old_error,torch.ones(1,dtype=torch.bool))
    assert bool((new_cos>old_cos).all()) and bool(((new_norm/target-1).abs()<=NORM_REL_TOL).all()) and bool((new_sse>old_error).all())
    assert not bool(accept.any()) and checks_sse['failed_source_SSE']==1
    # Row chunks change temporary memory only, not fields or the original r0.
    source=torch.randn(17,128);book=torch.randn(16,4).half();fields=initial(source,book,1.037)
    out1=recode_key_postres_direction(source,book,*fields,gain_factor=1.037,row_chunk=1)
    out2=recode_key_postres_direction(source,book,*fields,gain_factor=1.037,row_chunk=17)
    assert all(torch.equal(a,b) for a,b in zip(out1[:3],out2[:3]))
    assert out1[3]['max_relative_final_norm_error']<=NORM_REL_TOL and out1[3]['optimized_field_bytes_after']==out1[3]['optimized_field_bytes_before']==17*20
    # A normalized direction with a slightly perturbed FP16 radius need not
    # improve SSE merely because its cosine improves; the explicit SSE guard
    # is independently checked on the real initialized random cases above.
    return dict(status='PASS',schema='trajkv.round8.postres-fixed-norm-cpu-check.v1',
        dense_single_real_proposal_cases=dense,incremental_vs_full_real_scores=moments,
        zero_target_and_strict_tie_bitwise_noop=True,initialization_negative_controls=3,
        norm_infeasible_guard=checks,overflow_guard=checks_overflow,
        cosine_up_norm_feasible_SSE_up_rejected_guard=checks_sse,
        row_chunk_field_identity=True,original_r0_anchored_once=True,
        same_packed16B_and_existing_gain_alpha=True,added_retained_tensor_bytes=0,signs_unchanged=True,
        source_sha256=sha(Path(__file__).with_name('round8_postres_direction.py')),
        checker_sha256=sha(__file__),GPU_used=False,seed=20261118,torch_version=torch.__version__,
        dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in ('round8_codec.py','round6_depth_codec.py','quantizers.py')},
        scope='CPU synthetic complete-final-vector cosine/norm/SSE arithmetic; no bank/model/native-mean bridge/task/timing experiment')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=controls()
    (args.output/'check.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',GPU_used=False,
        dense_cases=6,real_conditional_scores=report['incremental_vs_full_real_scores'],report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',dense_cases=6,real_conditional_scores=report['incremental_vs_full_real_scores'],GPU_used=False)))


if __name__=='__main__':main()
