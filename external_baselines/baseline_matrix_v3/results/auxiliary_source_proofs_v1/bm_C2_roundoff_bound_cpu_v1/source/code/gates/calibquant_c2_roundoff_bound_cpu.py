"""CPU forward-error audit of an actual one-layer C2 CUDA source replay.

This checks each captured source reduction against its FP32/FP16 arithmetic
budget, then applies the exact triangle inequality to probability perturbations
and separately checked affine-cancellation roundoff. No score or tolerance is
fitted to the final output error.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    source_report_path=args.capture.parent/'complete.json'
    report=json.loads(source_report_path.read_text(encoding='utf-8'))
    assert report['status']=='SOURCE_OPERATOR_REPLAY_CAPTURED'
    assert report['production_source_replay_bitwise_equal'] is True
    assert report['evidence']['sha256']==sha(args.capture)
    x=torch.load(args.capture,map_location='cpu',weights_only=False)
    assert x['source_capture_sha256']==report['source_capture_sha256']
    u32=2.0**-24;u16=2.0**-11
    def gamma(n):
        assert n*u32<1
        return n*u32/(1-n*u32)
    def r16(value):return u16*value.abs()+2.0**-25
    checks=[]
    def check(name,error,bound):
        guard=1e-11*(1+bound.abs())
        violations=(error>bound+guard)|~torch.isfinite(error)|~torch.isfinite(bound)
        checks.append({'name':name,'violations':int(violations.sum()),'elements':error.numel(),
                       'max_actual_error':float(error.max()),'max_allowed_bound':float(bound.max())})
        return not bool(violations.any())
    def largest(tensor):
        remaining=int(tensor.flatten().argmax());coords=[]
        for width in reversed(tensor.shape):coords.append(remaining%width);remaining//=width
        return tuple(reversed(coords))
    q,ks,kz,vs,vz=(x[name] for name in ('q','k_scale','k_zero','v_scale','v_zero'))
    kcodes=x['k_codes'].double();vcodes=x['v_codes'].double();scale=x['scaling']
    scaled=(q*ks).double()
    exact_qk=scaled@kcodes.transpose(-1,-2)
    qk_sum_bound=gamma(q.shape[-1]-1)*(scaled.abs()@kcodes.abs().transpose(-1,-2))
    qk_bound=qk_sum_bound+r16(exact_qk.abs()+qk_sum_bound)
    check('official_QK_FP32_reduction_then_FP16_store',
          (x['raw_qk_source'].double()-exact_qk).abs(),qk_bound)
    products=x['bias_products_half'].double()
    exact_bias_sum=products.sum(-1,keepdim=True)
    bias_sum_bound=gamma(products.shape[-1]-1)*products.abs().sum(-1,keepdim=True)
    check('declared_FP32_bias_sum',(x['bias_sum_float32'].double()-exact_bias_sum).abs(),bias_sum_bound)
    bias_wide=exact_bias_sum*scale
    # The Python sqrt(D) denominator is first represented in FP32, then the
    # division rounds in FP32. Account for both operations before Half cast.
    bias_bound=bias_sum_bound*abs(scale)/(1-u32)+gamma(2)*bias_wide.abs()
    check('declared_FP32_bias_division_then_FP16_cast',
          (x['bias_half'].double()-bias_wide).abs(),bias_bound+r16(bias_wide.abs()+bias_bound))
    nv=x['visual_tokens'];text_values=x['text_values'].double();s=vs.double();z=vz.double()
    reconstructed_visual=s*(vcodes-z)
    all_values=torch.cat((reconstructed_visual,text_values),-2)
    pa=x['actual_probabilities_half'].double();pi=x['ideal_probabilities_half'].double()
    probability_bound=(pa-pi).abs()@all_values.abs()
    exact_outputs={};round_budgets={};components={}
    for role,p in (('actual',pa),('ideal',pi)):
        prefix='' if role=='actual' else 'ideal_'
        pvis=p[...,:nv];ptext=p[...,nv:]
        exact_C=pvis@vcodes;exact_P=pvis.sum(-1,keepdim=True)
        C=x['raw_wv_source' if role=='actual' else 'ideal_raw_wv_half'].double()
        P=x['weight_sum_half' if role=='actual' else 'ideal_weight_sum_half'].double()
        C_sum_bound=gamma(nv-1)*exact_C.abs()
        C_allowed=C_sum_bound+r16(exact_C.abs()+C_sum_bound) if role=='actual' else r16(exact_C)
        E_C=(C-exact_C).abs()
        check(role+'_WV_code_reduction_and_cast',E_C,C_allowed)
        P_sum_bound=gamma(nv-1)*exact_P
        E_P=(P-exact_P).abs()
        check(role+'_visual_probability_sum_and_cast',E_P,P_sum_bound+r16(exact_P+P_sum_bound))
        if role=='actual':
            left=x['affine_left_half'].double();coefficient=x['affine_coefficient_half'].double()
            right=x['affine_right_half'].double();visual=x['visual_output_half'].double()
            text=x['text_output_half'].double();output=x['actual_output'].transpose(1,2).double()
        else:
            left=(C.half()*vs).double();coefficient=(vs*vz).double()
            right=(coefficient.half()*P.half()).double()
            visual=x['ideal_visual_output_half'].double();text=x['ideal_text_output_half'].double()
            output=x['FP64_instrument_output'].transpose(1,2).double()
        e_left=(left-s*C).abs();e_coefficient=(coefficient-s*z).abs()
        e_right=(right-coefficient*P).abs();e_visual=(visual-(left-right)).abs()
        check(role+'_Half_scale_times_weighted_codes',e_left,r16(s*C))
        check(role+'_Half_scale_times_zero',e_coefficient,r16(s*z))
        check(role+'_Half_affine_bias_product',e_right,r16(coefficient*P))
        check(role+'_Half_affine_subtraction',e_visual,r16(left-right))
        exact_text=ptext@text_values
        E_text=(text-exact_text).abs()
        text_sum_bound=gamma(text_values.shape[-2]-1)*(ptext.abs()@text_values.abs())
        check(role+'_text_matmul_FP32_accumulation_FP16_store',E_text,
              text_sum_bound+r16(exact_text.abs()+text_sum_bound))
        e_final=(output-(visual+text)).abs()
        check(role+'_Half_visual_plus_text',e_final,r16(visual+text))
        # Each term is an observed intermediate operation error whose allowed
        # floating-point bound was checked above, not a final-output fudge.
        budget=s.abs()*E_C+e_left+(s*z).abs()*E_P+P.abs()*e_coefficient+e_right+e_visual+E_text+e_final
        exact_output=s*exact_C-s*z*exact_P+exact_text
        check(role+'_component_forward_error_closure',(output-exact_output).abs(),budget)
        exact_outputs[role]=exact_output;round_budgets[role]=budget
        components[role]={'max_weighted_code_roundoff':float((s.abs()*E_C).max()),
                          'max_probability_sum_roundoff':float(((s*z).abs()*E_P).max()),
                          'max_affine_intermediate_roundoff':float((e_left+P.abs()*e_coefficient+e_right+e_visual).max()),
                          'max_text_accumulation_roundoff':float(E_text.max()),'max_final_add_roundoff':float(e_final.max()),
                          'max_total_roundoff_budget':float(budget.max())}
    check('exact_weighted_probability_perturbation_inequality',
          (exact_outputs['actual']-exact_outputs['ideal']).abs(),probability_bound)
    actual=x['actual_output'].transpose(1,2).double();ideal=x['FP64_instrument_output'].transpose(1,2).double()
    gap=(actual-ideal).abs();total_bound=probability_bound+round_budgets['actual']+round_budgets['ideal']
    check('actual_vs_FP64_complete_forward_error_bound',gap,total_bound)
    where=largest(gap);nonzero=gap>0
    all_pass=all(item['violations']==0 for item in checks)
    result={'status':'PASS_FROZEN_CAPTURE_FORWARD_ERROR_AUDIT' if all_pass else 'SOURCE_ARITHMETIC_CONTRACT_NOT_VERIFIED',
        'scope':'This frozen operator capture only; no all-input theorem or task-quality claim',
        'GPU_used':False,'model_weights_loaded':False,'quality_scored':False,
        'source_capture_sha256':sha(args.capture),'source_replay_report_sha256':sha(source_report_path),
        'production_source_replay_bitwise_equal':True,
        'contract':{'QK_WV':'source-explicit FP32 products/reduction, FP16 store',
                    'bias':'FP16 products; FP32 sum/div then FP16 cast',
                    'text_matmul':'captured operation is checked against FP32 accumulation plus FP16-store bound; this is not asserted universally for all cublas algorithms',
                    'pointwise':'IEEE binary16 finite arithmetic, relative rounding bound u16 plus half-min-subnormal/2',
                    'probabilities':'actual and independent-instrument post-softmax FP16 probabilities are used directly; no unverified expf accuracy assumption',
                    'future_scope':'A different source/runtime/shape requires separate evidence; no output-ULP threshold is increased'},
        'checks':checks,'component_budgets':components,
        'probability_delta':{'max_L1':float((pa-pi).abs().sum(-1).max()),
                             'mean_L1':float((pa-pi).abs().sum(-1).mean()),
                             'max_channel_weighted_bound':float(probability_bound.max())},
        'observed_output':{'max_abs':float(gap.max()),'different_elements':int(nonzero.sum()),'elements':gap.numel(),
                           'relative_L2':float(gap.norm()/ideal.norm()),'max_gap_location_B_H_Q_D':list(where),
                           'at_max_gap':{'actual':float(actual[where]),'ideal':float(ideal[where]),
                              'probability_term':float(probability_bound[where]),
                              'actual_roundoff_budget':float(round_budgets['actual'][where]),
                              'ideal_roundoff_budget':float(round_budgets['ideal'][where]),
                              'total_allowed':float(total_bound[where])}},
        'max_observed_to_complete_bound_ratio':float((gap[nonzero]/total_bound[nonzero].clamp_min(1e-30)).max()) if bool(nonzero.any()) else 0.,
        'audit_source_sha256':sha(__file__),'created_unix':time.time()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'checks':len(checks),'max_abs':float(gap.max()),
                      'at_max_gap':result['observed_output']['at_max_gap']}))


if __name__=='__main__':main()
