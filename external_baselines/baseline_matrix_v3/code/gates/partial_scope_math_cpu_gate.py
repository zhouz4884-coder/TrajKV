"""Finite-distribution checks of the partial-scope attention identities."""
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.manual_seed(20261120);torch.set_num_threads(2)
    cases=[]
    for seed in range(12):
        z=torch.randn(11,dtype=torch.float64);t=torch.randn(5,dtype=torch.float64)
        e=torch.randn_like(z)*(seed+1)/3;v=torch.randn(11,7,dtype=torch.float64);u=torch.randn(5,7,dtype=torch.float64)
        vh=v+torch.randn_like(v)*.2
        pi=z.softmax(0);tau=t.softmax(0);ph=(z+e).softmax(0)
        zv=torch.logsumexp(z,0);zt=torch.logsumexp(t,0);zh=torch.logsumexp(z+e,0)
        beta=e.mean();eta=e-beta;moment=torch.logsumexp(z-zv+eta,0)
        s=torch.sigmoid(zv-zt);sh=torch.sigmoid(zh-zt)
        m=pi@v;mh=ph@vh;mt=tau@u
        o=s*m+(1-s)*mt;oh=sh*mh+(1-sh)*mt
        torch.testing.assert_close(zh-zv,beta+moment,atol=2e-14,rtol=2e-14)
        torch.testing.assert_close(oh-o,(sh-s)*(m-mt)+sh*(mh-m),atol=2e-14,rtol=2e-14)
        tv=(ph-pi).abs().sum()/2;bound=torch.tanh((eta.max()-eta.min())/4)
        assert tv<=bound+1e-14
        selected=torch.arange(0,11,2)
        torch.testing.assert_close(torch.logsumexp(z[selected],0)-zv,torch.log(pi[selected].sum()),atol=2e-14,rtol=2e-14)
        cases.append({'seed':seed,'TV':float(tv),'TV_bound':float(bound)})
    R=torch.tensor(4.,dtype=torch.float64)
    pi=torch.stack((1/(1+torch.exp(R/2)),torch.exp(R/2)/(1+torch.exp(R/2))))
    eta=torch.stack((R/2,-R/2));ph=(pi.log()+eta).softmax(0)
    torch.testing.assert_close((ph-pi).abs().sum()/2,torch.tanh(R/4),atol=2e-14,rtol=2e-14)
    note=ROOT/'docs/ANALYSIS_partial_scope_attention_v1.md'
    result={'status':'PASS_CPU_PARTIAL_SCOPE_ALGEBRA_REVIEW','GPU_used':False,'real_model_inference':False,
        'note_sha256':hashlib.sha256(note.read_bytes()).hexdigest(),'random_finite_cases':cases,
        'two_point_TV_bound_tight':True,'TV_definition':'half L1',
        'mathematical_errors_found':False,
        'scope_requirements':['fixed q','text logits/values unchanged','nonempty active visual/text sets and finite partition functions',
                              'eta zero under uniform mean, not necessarily pi-weighted mean','causal text masking per query',
                              'empty pruning/text-zero cases interpreted with limits; conditional pi/tau may be undefined'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'audit/partial_scope_math_review_v1.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'finite_cases':len(cases),'math_errors_found':False}))


if __name__=='__main__':main()
