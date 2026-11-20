"""Explicit KIVI-axis visual-prefix adapter, with actual packed storage.

R0 is the project's KIVI-style technical anchor. R128 retains the published
prefill K partial-group and V residual-window policy on the visual segment;
text is exact and the visual prefix is immutable, so neither is advertised
as the original whole-text-and-image rolling Llama KIVI system.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import torch

UPSTREAM_REPO='https://github.com/jy-yuan/KIVI'
UPSTREAM_COMMIT='876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6'
_CODEC=None


def upstream_codec():
    global _CODEC
    if _CODEC is None:
        path=Path(__file__).resolve().parents[2]/'third_party/kivi/quant/new_pack.py'
        spec=importlib.util.spec_from_file_location('matrix_kivi_source_new_pack',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_CODEC=module
    return _CODEC


def unpack(code,bits):
    values=(code.long()&0xffffffff).unsqueeze(-1)
    shifts=torch.arange(32//bits,device=code.device,dtype=torch.int64)*bits
    return ((values>>shifts)&((1<<bits)-1)).reshape(*code.shape[:-1],-1)


def dequant(code,scale,minimum,bits):
    levels=unpack(code,bits).reshape(*scale.shape,128)
    return ((levels.to(scale.dtype)*scale[...,None])+minimum[...,None]).reshape(*code.shape[:-1],-1)


@dataclass
class KIVIVisualState:
    k_code:torch.Tensor
    k_scale:torch.Tensor
    k_min:torch.Tensor
    v_code:torch.Tensor
    v_scale:torch.Tensor
    v_min:torch.Tensor
    k_tail:torch.Tensor
    v_tail:torch.Tensor
    positions:torch.Tensor
    bits:int
    nv:int
    nkq:int
    nvq:int
    residual_length:int
    backend:str='triton_packed'

    def named_tensors(self,include_constants=True):
        return {name:getattr(self,name) for name in ('k_code','k_scale','k_min','v_code','v_scale','v_min','k_tail','v_tail','positions')}
    owner_tensors=named_tensors

    def materialize(self,dtype=None):
        kh=(dequant(self.k_code,self.k_scale,self.k_min,self.bits).transpose(-1,-2).contiguous()
            if self.nkq else self.k_tail[:,:,:0])
        vh=(dequant(self.v_code,self.v_scale,self.v_min,self.bits) if self.nvq else self.v_tail[:,:,:0])
        key=torch.cat((kh,self.k_tail),-2);value=torch.cat((vh,self.v_tail),-2)
        return key.to(dtype or key.dtype),value.to(dtype or value.dtype)

    def attend(self,q,*,exact_keys,exact_values,exact_positions,query_positions,scale=None,backend=None):
        selected=backend or self.backend
        if selected=='triton_packed':
            from methods.kivi_reader import attention
            return attention(q,self,exact_keys,exact_values,exact_positions,query_positions,scale)
        if selected!='torch_dequant_reference':raise ValueError('Explicit KIVI reader backend required')
        k,v=self.materialize();h=q.shape[1];groups=h//k.shape[1]
        keys=torch.cat((k,exact_keys),-2).repeat_interleave(groups,1)
        values=torch.cat((v,exact_values),-2).repeat_interleave(groups,1)
        pos=torch.cat((self.positions,exact_positions));scores=(q.float()@keys.float().transpose(-1,-2))*(scale or q.shape[-1]**-.5)
        scores=scores.masked_fill(pos[None,None,None,:]>query_positions[None,None,:,None],-torch.inf)
        weights=torch.softmax(scores,-1).half()
        return (weights.float()@values.float()).to(q.dtype)

    def owner_report(self):
        tensors=self.named_tensors();seen=set();total=0
        for t in tensors.values():
            storage=t.untyped_storage();key=(str(t.device),storage.data_ptr(),storage.nbytes())
            if key not in seen:seen.add(key);total+=storage.nbytes()
        denominator=2*self.k_tail.shape[1]*self.nv*128*2
        return dict(method='KIVI-style visual port',upstream_commit=UPSTREAM_COMMIT,bits_nominal=self.bits,
            reader_backend=self.backend,stored_visual_KV_bytes=total,positions_bytes=self.positions.untyped_storage().nbytes(),
            partial_K_tail_tokens=self.nv-self.nkq,V_residual_tail_tokens=self.nv-self.nvq,
            visual_residual_length=self.residual_length,original_full_KV_rolling_system=False,
            all_owner_bytes_including_constants=total,effective_bits_including_constants=16*total/denominator,
            temporary_full_visual_FP16=self.backend=='torch_dequant_reference')


class KIVIStyleAdapter:
    def __init__(self,bits=2,group_size=128,residual_length=0,reader_backend='triton_packed'):
        if bits not in (2,4) or group_size!=128 or residual_length not in (0,128):
            raise ValueError('Locked legal KIVI-style widths2/4,g128,R0/R128 only')
        self.bits,self.group_size,self.residual_length,self.backend=bits,group_size,residual_length,reader_backend

    @torch.no_grad()
    def encode(self,k,v,positions,layer_idx=0):
        if k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=1 or k.shape[-1]!=128 or k.dtype!=torch.float16:
            raise ValueError('Actual batch1 post-RoPE FP16 K/V D128 required')
        n=k.shape[-2];nkq=n-n%128;nvq=max(0,n-self.residual_length);h=k.shape[1]
        quant=upstream_codec().triton_quantize_and_pack_along_last_dim
        kc,ks,km=(quant(k[:,:,:nkq].transpose(-1,-2).contiguous(),128,self.bits) if nkq else
            (torch.empty((1,h,128,0),device=k.device,dtype=torch.int32),torch.empty((1,h,128,0),device=k.device,dtype=k.dtype),torch.empty((1,h,128,0),device=k.device,dtype=k.dtype)))
        vc,vs,vm=(quant(v[:,:,:nvq].contiguous(),128,self.bits) if nvq else
            (torch.empty((1,h,0,128//(32//self.bits)),device=k.device,dtype=torch.int32),torch.empty((1,h,0,1),device=k.device,dtype=k.dtype),torch.empty((1,h,0,1),device=k.device,dtype=k.dtype)))
        state=KIVIVisualState(kc,ks,km,vc,vs,vm,k[:,:,nkq:].detach().contiguous().clone(),
            v[:,:,nvq:].detach().contiguous().clone(),positions.detach(),self.bits,n,nkq,nvq,self.residual_length,self.backend)
        if any(t.untyped_storage().data_ptr() in (k.untyped_storage().data_ptr(),v.untyped_storage().data_ptr())
               for name,t in state.named_tensors().items() if name!='positions' and t.numel()):
            raise RuntimeError('KIVI state retains full source storage')
        return state
