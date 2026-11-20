"""CPU trace of the largest C2 source-versus-FP64-instrument difference."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2)
    x=torch.load(args.capture,map_location='cpu',weights_only=False)
    actual=x['actual_output'].transpose(1,2);ideal=x['FP64_instrument_output'].transpose(1,2)
    gap=(actual.double()-ideal.double()).abs();flat=int(gap.flatten().argmax());shape=gap.shape;coords=[]
    for width in reversed(shape):coords.append(flat%width);flat//=width
    b,h,t,d=reversed(coords)
    nv=x['visual_tokens'];p=x['actual_probabilities_half'][b,h,t,:nv].double();q=x['ideal_probabilities_half'][b,h,t,:nv].double()
    codes=x['v_codes'][b,h,:,d].double();wide_A=float(p@codes);wide_I=float(q@codes)
    local=(b,h,t,d);value_scale=float(x['v_scale'][b,h,0,d]);zero=float(x['v_zero'][b,h,0,d])
    C_A=float(x['raw_wv_source'][local]);C_I=float(x['ideal_raw_wv_half'][local])
    P_A=x['weight_sum_half'][b,h,t,0];P_I=x['ideal_weight_sum_half'][b,h,t,0]
    s=x['v_scale'][b,h,0,d];z=x['v_zero'][b,h,0,d]
    left_I=float(x['ideal_raw_wv_half'][local]*s);coef=float(s*z);right_I=float((s*z)*P_I)
    stages={
        'weighted_code_exact_FP64_from_own_p':(wide_A,wide_I),
        'weighted_code_stored_Half':(C_A,C_I),
        'visual_weight_sum_Half':(float(P_A),float(P_I)),
        'scale_times_weighted_code_Half':(float(x['affine_left_half'][local]),left_I),
        'scaled_zero_times_weight_sum_Half':(float(x['affine_right_half'][local]),right_I),
        'visual_output_Half':(float(x['visual_output_half'][local]),float(x['ideal_visual_output_half'][local])),
        'text_output_Half':(float(x['text_output_half'][local]),float(x['ideal_text_output_half'][local])),
        'final_output_Half':(float(actual[local]),float(ideal[local]))}
    midpoint=(C_A+C_I)/2
    result={'status':'CPU_MAX_GAP_LOCAL_TRACE_COMPLETE','GPU_used':False,'quality_scored':False,
        'source_capture_sha256':hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        'location_B_H_Q_D':[b,h,t,d],'visual_value_scale':value_scale,'visual_zero_point':zero,'stored_scale_times_zero':coef,
        'stages':{key:{'actual_path':a,'FP64_instrument_path':i,'difference':a-i} for key,(a,i) in stages.items()},
        'weighted_code_half_rounding':{'stored_values_midpoint':midpoint,'actual_exact_distance_to_midpoint':wide_A-midpoint,
                                     'ideal_exact_distance_to_midpoint':wide_I-midpoint,
                                     'exact_values_straddle_midpoint':(wide_A-midpoint)*(wide_I-midpoint)<=0,
                                     'actual_kernel_vs_FP64_rounded_same_actual_p_difference':C_A-float(x['raw_wv_FP64_same_actual_probabilities'][local])},
        'probability_L1':float((p-q).abs().sum()),
        'scope':'One frozen operator coordinate only; trace the numerical path, not a task-quality conclusion'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
