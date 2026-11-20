"""Visual port of pinned KIVI example.py b2/g32/R32 prefill policy.

The original source codec is called unchanged on CUDA. Text remains FP16 and
the visual prefix is immutable; this is not the original rolling Llama system.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import torch

UPSTREAM_REPO='https://github.com/jy-yuan/KIVI'
UPSTREAM_COMMIT='876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6'
METHOD_ID='KIVI_source_example_K2V2_g32_R32_visual'
_CODEC=None


def source_codec():
    global _CODEC
    if _CODEC is None:
        path=Path(__file__).resolve().parents[2]/'third_party/kivi/quant/new_pack.py'
        digest=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if digest!='3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e':raise RuntimeError('Official KIVI encoder source changed')
        spec=importlib.util.spec_from_file_location('matrix_kivi_example_source_codec',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_CODEC=module
    return _CODEC


def prefill_lengths(n,residual_length=32):
    if n<=0 or residual_length!=32:raise ValueError('Positive visual length and source-example R32 required')
    # The source uses residual_length for K's prefill split, NOT group_size.
    return n-n%residual_length,max(0,n-residual_length)


def unpack(code,bits):
    packed=(code.long()&0xffffffff).unsqueeze(-1)
    shifts=torch.arange(32//bits,device=code.device,dtype=torch.int64)*bits
    return ((packed>>shifts)&((1<<bits)-1)).reshape(*code.shape[:-1],-1)


def dequant(code,scale,minimum,bits,group_size):
    levels=unpack(code,bits).reshape(*scale.shape,group_size)
    return ((levels.to(scale.dtype)*scale[...,None])+minimum[...,None]).reshape(*code.shape[:-1],-1)


@dataclass
class KIVIExampleState:
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
    group_size:int=32
    residual_length:int=32
    backend:str='triton_packed'

    def named_tensors(self,include_constants=True):
        return {name:getattr(self,name) for name in ('k_code','k_scale','k_min','v_code','v_scale','v_min','k_tail','v_tail','positions')}
    owner_tensors=named_tensors

    def materialize(self,dtype=None):
        kh=dequant(self.k_code,self.k_scale,self.k_min,self.bits,self.group_size).transpose(-1,-2).contiguous() if self.nkq else self.k_tail[:,:,:0]
        vh=dequant(self.v_code,self.v_scale,self.v_min,self.bits,self.group_size) if self.nvq else self.v_tail[:,:,:0]
        keys,values=torch.cat((kh,self.k_tail),-2),torch.cat((vh,self.v_tail),-2)
        return keys.to(dtype or keys.dtype),values.to(dtype or values.dtype)

    def attend(self,q,*,exact_keys,exact_values,exact_positions,query_positions,scale=None,backend=None):
        selected=backend or self.backend
        if selected=='triton_packed':
            from methods.kivi_example_reader import attention
            return attention(q,self,exact_keys,exact_values,exact_positions,query_positions,scale)
        if selected!='torch_dequant_reference':raise ValueError('Declare an explicit reader backend; no silent dequant fallback')
        k,v=self.materialize();groups=q.shape[1]//k.shape[1]
        keys=torch.cat((k,exact_keys),-2).repeat_interleave(groups,1)
        values=torch.cat((v,exact_values),-2).repeat_interleave(groups,1)
        positions=torch.cat((self.positions,exact_positions))
        factor=q.shape[-1]**-.5 if scale is None else float(scale)
        logits=(q.float()@keys.float().transpose(-1,-2))*factor
        logits=logits.masked_fill(positions[None,None,None,:]>query_positions[None,None,:,None],-torch.inf)
        weights=torch.softmax(logits,-1).half()
        return (weights.float()@values.float()).to(q.dtype)

    def owner_report(self):
        tensors=self.named_tensors();seen=set();charges={}
        for name,t in tensors.items():
            storage=t.untyped_storage();key=(str(t.device),storage.data_ptr(),storage.nbytes())
            charges[name]=0 if key in seen else storage.nbytes();seen.add(key)
        total=sum(charges.values());denominator=2*self.k_tail.shape[0]*self.k_tail.shape[1]*self.nv*128*2
        return {'method':METHOD_ID,'upstream_repo':UPSTREAM_REPO,'upstream_commit':UPSTREAM_COMMIT,
            'method_paper':'https://arxiv.org/abs/2402.02750','source_configuration':'pinned example.py k_bits2/v_bits2/group_size32/residual_length32',
            'faithfulness_status':'SOURCE_EXAMPLE_HYPERPARAMETERS_AND_PREFILL_POLICY_VISUAL_SCOPE_PORT',
            'original_full_KV_rolling_system':False,'text_KV':'exact FP16 outside visual owner','reader_backend':self.backend,
            'bits_nominal':self.bits,'group_size':self.group_size,'visual_residual_length':self.residual_length,
            'K_prefill_split_modulus':'residual_length','partial_K_tail_tokens':self.nv-self.nkq,'V_residual_tail_tokens':self.nv-self.nvq,
            'packed_payload_bytes':sum(charges[n] for n in ('k_code','v_code')),
            'coder_metadata_bytes':sum(charges[n] for n in ('k_scale','k_min','v_scale','v_min')),
            'full_precision_visual_tail_bytes':sum(charges[n] for n in ('k_tail','v_tail')),
            'positions_bytes':charges['positions'],'encoder_constants_bytes':0,'lookup_bytes':0,
            'stored_visual_KV_bytes':total,'all_owner_bytes_including_constants':total,'full_kv_byte_denominator':denominator,
            'actual_effective_visual_bits':16*total/denominator,'effective_bits_including_constants':16*total/denominator,
            'V_metadata_groups_per_D128_token':128//self.group_size,'temporary_full_visual_FP16':self.backend=='torch_dequant_reference'}


class KIVIExampleAdapter:
    def __init__(self,bits=2,group_size=32,residual_length=32,reader_backend='triton_packed'):
        if (bits,group_size,residual_length)!=(2,32,32):raise ValueError('Only the pinned source-example b2/g32/R32 point is admitted')
        if reader_backend not in ('triton_packed','torch_dequant_reference'):raise ValueError('Explicit reader backend required')
        self.bits,self.group_size,self.residual_length,self.backend=bits,group_size,residual_length,reader_backend

    @torch.no_grad()
    def encode(self,k,v,positions,layer_idx=0):
        if k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=1 or k.shape[-1]!=128 or k.dtype!=torch.float16 or v.dtype!=k.dtype or v.device!=k.device or not k.is_cuda:
            raise ValueError('Production encoder requires CUDA batch1 post-RoPE FP16 K/V D128')
        n,h=k.shape[-2],k.shape[1];nkq,nvq=prefill_lengths(n,self.residual_length)
        if positions.ndim!=1 or len(positions)!=n or positions.dtype not in (torch.int32,torch.int64):raise ValueError('One original integer position per visual token required')
        if not torch.equal(positions,positions.sort().values) or positions.unique().numel()!=n or int(positions.min())<0:raise ValueError('Original positions must be sorted unique nonnegative')
        quant=source_codec().triton_quantize_and_pack_along_last_dim;g=self.group_size;b=self.bits
        kc,ks,km=quant(k[:,:,:nkq].transpose(-1,-2).contiguous(),g,b) if nkq else (
            torch.empty((1,h,128,0),device=k.device,dtype=torch.int32),k.new_empty((1,h,128,0)),k.new_empty((1,h,128,0)))
        vc,vs,vm=quant(v[:,:,:nvq].contiguous(),g,b) if nvq else (
            torch.empty((1,h,0,128//(32//b)),device=k.device,dtype=torch.int32),k.new_empty((1,h,0,128//g)),k.new_empty((1,h,0,128//g)))
        state=KIVIExampleState(kc,ks,km,vc,vs,vm,k[:,:,nkq:].detach().contiguous().clone(),
            v[:,:,nvq:].detach().contiguous().clone(),positions.detach().to(device=k.device,dtype=torch.int64).contiguous().clone(),
            b,n,nkq,nvq,g,self.residual_length,self.backend)
        forbidden={k.untyped_storage().data_ptr(),v.untyped_storage().data_ptr()}
        if any(t.numel() and t.untyped_storage().data_ptr() in forbidden for t in state.named_tensors().values()):raise RuntimeError('Visual owner retains full source storage')
        return state
