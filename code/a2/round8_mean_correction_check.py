"""CPU mathematical/packed-field fixtures; no model, SSH or GPU execution."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import decode_key,pack_nibbles
from round6_depth_codec import decode_sign_stage
from round8_mean_correction import (wht128,final_packed_key_mean,compensated_mean_header,
                                    correct_state_mean_inplace,mean_corrected_bank_class)


def digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def fields(state):
    return {key:digest(value) for key,value in vars(state).items()
            if isinstance(value,torch.Tensor) and key!='direction_mean'}


def bytes_owned(states,book):
    tensors=[book]+[v for state in states for v in vars(state).values() if isinstance(v,torch.Tensor)]
    seen={}
    for t in tensors:
        seen[t.untyped_storage().data_ptr()]=t.untyped_storage().nbytes()
    return dict(total_tensor_storage_bytes=sum(seen.values()),
                mean_slot_bytes=sum(s.direction_mean.numel()*2 for s in states))


def fixture(shape,seed=18):
    torch.manual_seed(seed);h,n=2,32
    book=(torch.randn(shape)*.4+.15).half()
    groups=128//shape[-1];words=shape[-2]
    index=torch.randint(words,(1,h,n,groups),dtype=torch.uint8)
    code=pack_nibbles(index) if words==16 else index
    state=SimpleNamespace(direction_code=code,direction_gain=(.4+torch.rand(1,h,n,1)).half(),
        direction_mean=(torch.randn(1,h,1,128)*.1).half(),visual_pos=torch.arange(n,dtype=torch.int64),
        depth_residual_lookup=torch.full((2,h,2),-1,dtype=torch.int32),
        depth_residual_code=torch.randint(-(2**31),2**31-1,(4,n,4),dtype=torch.int32),
        depth_residual_alpha=(torch.rand(4,n)*.04).half(),
        v_code=torch.randint(-(2**31),2**31-1,(1,h,128,n//32),dtype=torch.int32),
        v_scale=torch.rand(1,h,128,1).half(),v_min=torch.randn(1,h,128,1).half(),
        text_k=torch.randn(1,h,3,128).half(),text_v=torch.randn(1,h,3,128).half())
    state.depth_residual_lookup[0,0]=torch.tensor([0,1],dtype=torch.int32)
    state.depth_residual_lookup[0,1,0]=2
    state.depth_residual_lookup[1,0,0]=3
    return state,book


def full_decode(state,book):
    final=decode_key(state.direction_code,state.direction_gain,book)
    for head in range(final.shape[1]):
        for stage in range(2):
            slot=int(state.depth_residual_lookup[0,head,stage])
            if slot>=0:
                final[0,head]=decode_sign_stage(final[0,head],state.depth_residual_code[slot],state.depth_residual_alpha[slot])
    return final


def softmax_checks():
    torch.manual_seed(20261118)
    q=torch.randn(3,128,dtype=torch.float64)
    k=torch.randn(7,128,dtype=torch.float64);v=torch.randn(7,5,dtype=torch.float64)
    delta=torch.randn(128,dtype=torch.float64)*.3
    old=q@k.T/math.sqrt(128);all_shift=q@(k+delta).T/math.sqrt(128)
    wa,wb=old.softmax(-1),all_shift.softmax(-1)
    assert torch.allclose(wa,wb,rtol=0,atol=1e-12)
    assert torch.allclose(wa@v,wb@v,rtol=0,atol=1e-12)
    split=4;new=old.clone();new[:,:split]=q@(k[:split]+delta).T/math.sqrt(128)
    logodds=lambda score:score[:,:split].logsumexp(-1)-score[:,split:].logsumexp(-1)
    expected=q@delta/math.sqrt(128)
    error=(logodds(new)-logodds(old)-expected).abs().max()
    assert float(error)<1e-12
    assert torch.allclose(new[:,:split].softmax(-1),old[:,:split].softmax(-1),rtol=0,atol=1e-12)
    visible=float((new.softmax(-1)@v-wa@v).abs().max())
    assert visible>1e-6
    x=torch.randn(2,128,dtype=torch.float64)
    inverse=float((wht128(wht128(x))-x).abs().max());assert inverse<1e-12
    return dict(all_key_common_translation_softmax=True,all_key_attention_output_invariant=True,
        visual_only_mixed_text_output_change_max=visible,visual_conditional_distribution_invariant=True,
        group_logodds_shift_max_abs_error=float(error),WHT_self_inverse_max_abs_error=inverse,
        scope='FP64 fixed-q mathematical controls; not GPU bitwise equivalence or whole-model causal attribution')


def packed_checks():
    reports=[]
    for shape in ((16,4),(256,8),(32,16,4)):
        state,book=fixture(shape);original=fields(state);pointer=state.direction_mean.data_ptr()
        owner=bytes_owned([state],book);mu0=state.direction_mean.clone();final=full_decode(state,book)
        mean64=final.double().mean(-2,keepdim=True)
        measured=final_packed_key_mean(state.direction_code,state.direction_gain,book,state.depth_residual_lookup[0],
            state.depth_residual_code,state.depth_residual_alpha,chunk=7)
        assert torch.equal(mean64,measured)
        stats=correct_state_mean_inplace(state,book,chunk=7)
        assert fields(state)==original and state.direction_mean.data_ptr()==pointer and bytes_owned([state],book)==owner
        after_mean=state.direction_mean.double()+wht128(mean64)
        residual=after_mean-mu0.double()
        assert float(residual.abs().max())<float((wht128(mean64)).abs().max())
        # The FP16 physical header change is the exact fixed-q visual group
        # score offset, even when the rotated query was rounded to FP16.
        torch.manual_seed(29);q=torch.randn(1,4,3,128).half().double();g=2
        qr=wht128(q.float()).half().double()
        body=qr@final.double().repeat_interleave(g,1).transpose(-1,-2)/math.sqrt(128)
        old_bias=(q*mu0.double().repeat_interleave(g,1)).sum(-1)/math.sqrt(128)
        new_bias=(q*state.direction_mean.double().repeat_interleave(g,1)).sum(-1)/math.sqrt(128)
        old_score=body+old_bias[...,None];new_score=body+new_bias[...,None]
        expected=(q*(state.direction_mean.double()-mu0.double()).repeat_interleave(g,1)).sum(-1)/math.sqrt(128)
        common_error=float((new_score-old_score-expected[...,None]).abs().max());assert common_error<1e-12
        text=q@state.text_k.double().repeat_interleave(g,1).transpose(-1,-2)/math.sqrt(128)
        odds=lambda vis:vis.logsumexp(-1)-text.logsumexp(-1)
        odds_error=float((odds(new_score)-odds(old_score)-expected).abs().max());assert odds_error<1e-12
        assert torch.allclose(old_score.softmax(-1),new_score.softmax(-1),rtol=0,atol=1e-12)
        reports.append(dict(book_shape=list(shape),K_stages=[2,1],real_Nv=32,chunked_final_FP16_mean_exact=True,
            all_non_mean_tensor_fingerprints_unchanged=True,header_pointer_preserved=True,owner_before=owner,
            owner_after=bytes_owned([state],book),header_bias_formula_max_abs_error=common_error,
            mixed_segment_logodds_formula_max_abs_error=odds_error,correction=stats))
    # No residual pool is needed when all K slots are absent.
    state,book=fixture((16,4));state.depth_residual_lookup[0].fill_(-1)
    del state.depth_residual_code,state.depth_residual_alpha
    stats=correct_state_mean_inplace(state,book,chunk=5)
    assert stats['added_persistent_tensor_bytes']==0
    return reports


def integration_check():
    class FixtureBase:
        settings={}
        def _initialize(self,states,book):self.states,self.book=states,book;return 'BASE_INITIALIZED'
        def owned(self):return bytes_owned(self.states,self.book)
        def result_metadata(self):return dict(backend='CPU fixture')
    cls=mean_corrected_bank_class(FixtureBase)
    state,book=fixture((16,4));before=fields(state);cls.settings={'visual_mean_correction':True};bank=cls()
    owner=bytes_owned([state],book)
    assert bank._initialize([state],book)=='BASE_INITIALIZED'
    assert bank.owned()==owner and fields(state)==before and bank.result_metadata()['visual_mean_correction'] is True
    assert not any(isinstance(v,torch.Tensor) for row in bank.mean_correction_statistics for v in row.values())
    control,book2=fixture((16,4),19);mu=control.direction_mean.clone();cls.settings={'visual_mean_correction':False};off=cls()
    assert off._initialize([control],book2)=='BASE_INITIALIZED' and torch.equal(mu,control.direction_mean)
    assert off.mean_correction_statistics==[]
    return dict(status='PASS',super_then_header_only_stub=True,disabled_identity=True,
        new_retained_tensor_fields=0,main_L28_H4_mean_bytes_before=28*4*128*2,
        main_L28_H4_mean_bytes_after=28*4*128*2,main_complete_cache_byte_delta=0,
        scope='CPU subclass fixture only; actual model/owner/reader integration is not GPU-validated')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('CPU check requires hidden CUDA')
    if a.output.exists():raise ValueError('Use new immutable CPU evidence output')
    result=dict(schema='trajkv.round8.mean-correction-cpu-check.v1',status='PASS',GPU_used=False,
        softmax=softmax_checks(),packed_fields=packed_checks(),integration=integration_check(),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('round8_mean_correction.py','round8_mean_correction_check.py','round8_codec.py','round6_depth_codec.py')},
        limitations='Mean targets stored FP16 mu0; add original source-mean quantization/reduction error for true mean. Rounded H(q), FP32 reductions, full-model feedback and task quality require later real-reader/model checks.',
        scope='Mathematics and synthetic CPU packed-state fixtures only; no model/SSH/GPU/dataset or task/speed result')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status='PASS',packed_layouts=len(result['packed_fields']),GPU_used=False,
        persistent_cache_byte_delta=0,output=str(a.output))))


if __name__=='__main__':main()
