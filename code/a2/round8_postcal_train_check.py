"""CPU synthetic field/dense/gradient/config checks; not actual GPU evidence."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import torch

from round8_postcal_train import dense_readout,bridge_forward,validate_config,sha,write_new,aggregate_kl
from round8_postcal_fields import postcal_fields,decode_postcal_visual_fields,identity_factors
from round8_mean_correction_check import fixture,full_decode
from round8_mean_correction import wht128
from round8_check import frozen_reference_functions
from round6_depth_codec import decode_sign_stage


def require(value,message):
    if not value:raise RuntimeError(message)


def cpu_fixture():
    state,book=fixture((16,4),20261119);state.v_min[0,0,0,0]=-0.
    f=identity_factors(2);candidate=postcal_fields(state,f)
    names=('direction_code','direction_gain','direction_mean','v_code','v_scale','v_min','depth_residual_lookup','depth_residual_code','depth_residual_alpha')
    require(all(torch.equal(getattr(state,n).view(torch.uint8),getattr(candidate,n).view(torch.uint8)) for n in names),'CPU theta0 changed physical bytes')
    k,v=decode_postcal_visual_fields(state,f,book,32);require(torch.equal(k,full_decode(state,book)),'CPU K final stage decode differs')
    decoder,_=frozen_reference_functions();oldv=decoder(state.v_code,state.v_scale,state.v_min,32,1).transpose(2,3).contiguous()
    for h in range(2):
        for st in range(2):
            slot=int(state.depth_residual_lookup[1,h,st])
            if slot>=0:oldv[0,h]=decode_sign_stage(oldv[0,h],state.depth_residual_code[slot],state.depth_residual_alpha[slot])
    require(torch.equal(v,oldv),'CPU binaryV/final stage decode differs')
    torch.manual_seed(1212);q=torch.randn(1,4,3,128).half().requires_grad_(True);projection=torch.randn(128,11)*.2
    teacher=torch.randn(1,3,11).log_softmax(-1);params=[torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    text_k=state.text_k;text_v=state.text_v;vp=torch.arange(32);tp=torch.arange(32,35)
    def loss():
        table=torch.stack((params[0].exp(),params[1].exp(),params[2]));new=postcal_fields(state,table)
        dk,dv=decode_postcal_visual_fields(state,table,book,32)
        dense=dense_readout(q,dk,dv,new.direction_mean,text_k,text_v,vp,tp,67,wht128)
        # CPU-only mock of an exact forward; no Triton/CUDA or VLM is implied.
        out=bridge_forward(dense,dense.detach().clone())
        lp=(out.float().mean(1)@projection).log_softmax(-1)
        return (teacher.exp()*(teacher-lp)).sum(-1).mean()
    optimizer=torch.optim.Adam(params,lr=.01);initial=float(loss().detach());norms=None;qnorm=None
    for step in range(16):
        optimizer.zero_grad(set_to_none=True);q.grad=None;value=loss();value.backward()
        current=[float(p.grad.norm()) for p in params]
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),'CPU missing/nonfinite family gradient')
        require(q.grad is not None and bool(torch.isfinite(q.grad).all()) and float(q.grad.float().norm())>0,'CPU q derivative detached/nonfinite/zero')
        if step==0:
            require(all(n>0 for n in current),'CPU first field family derivative zero');norms=current;qnorm=float(q.grad.float().norm())
        optimizer.step()
    final=float(loss().detach());require(final<initial,'CPU synthetic surrogate fixture KL failed to decrease')
    return dict(theta0_all_field_bytes_exact=True,K_and_V_actual_nested_FP16_decoders_exact=True,
        first_family_gradient_norms=norms,first_query_gradient_norm=qnorm,synthetic_initial_KL=initial,synthetic_final_KL=final,
        synthetic_updates=16,actual_GPU_or_VLM_gradient_executed=False,
        scope='Small CPU synthetic shadow/backward fixture only; actual first-Q model and full70Q train gates remain mandatory')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=json.loads(a.config.read_text());contract=validate_config(cfg);controls=cpu_fixture()
    for change in ('steps','objective','cap','stage'):
        bad=copy.deepcopy(cfg)
        if change=='steps':bad['steps']=201
        elif change=='objective':bad['objective']='native_MSE'
        elif change=='cap':bad['arguments']['max_patches']=16
        else:bad['bank_settings']['res_stages_limit']=43
        try:validate_config(bad)
        except RuntimeError:pass
        else:raise RuntimeError('Invalid training design accepted: '+change)
    command=[sys.executable,str(Path(__file__).with_name('round8_postcal_train.py')),'--config',str(a.config),'--output',str(a.output/'unused_GPU_output'),'--plan-only']
    process=subprocess.run(command,text=True,capture_output=True,check=True);plan=json.loads(process.stdout)
    require(plan['status']=='PLAN_ONLY' and plan['questions']==70 and plan['steps']==200 and plan['GPU_executed'] is False,'Actual trainer CLI plan mismatch')
    result=dict(status='PASS',schema='trajkv.round8.postcal-trainer-CPU-check.v1',config_sha256=sha(a.config),checker_sha256=sha(__file__),
        trainer_sha256=sha(Path(__file__).with_name('round8_postcal_train.py')),field_helper_sha256=sha(Path(__file__).with_name('round8_postcal_fields.py')),
        contract=contract,synthetic_gradient=controls,plan=plan,invalid_configs_rejected=4,GPU_used=False,SSH_used=False,
        actual_first_model_gradient_pending=True,actual_full70Q_train_KL_decrease_pending=True)
    write_new(a.output/'check.json',result);write_new(a.output/'complete.json',dict(status='PASS',check_sha256=sha(a.output/'check.json'),GPU_used=False))
    print(json.dumps(dict(status='PASS',GPU_used=False,config_plan=True,synthetic_gradient=controls,full_priority_bytes=contract['full_priority_bytes'])),flush=True)


if __name__=='__main__':main()
