"""TurboQuant-MSE / OnlyTerp primary-source visual variant; QJL is OFF.

Uses OnlyTerp's unmodified turboquant_mse_encode/decode aliases, original
FP32 vector math, FP16 norms, K-derived variance groups and Hadamard padding.
Only categorical indices/maps/signs are losslessly packed. This is neither
TurboQuant-prod nor a reproduction of the paper's mislabeled 2.5-bit table.

The explicit reference reader expands packed data to temporary FP32 K/V.
It is a real compressed persistent owner, not an optimized paper kernel.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REPO = 'https://github.com/OnlyTerp/turboquant'
UPSTREAM_COMMIT = '1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306'
PAPER = 'https://arxiv.org/abs/2504.19874v1'
READER = 'torch_materialize_fp32_reference'
SOURCE_LOCK = 'configs/methods/onlyterp_mse_source_lock_v2.json'
_SOURCE = None


def _source():
    global _SOURCE
    if _SOURCE is None:
        lock = json.loads((ROOT / SOURCE_LOCK).read_text(encoding='utf-8'))
        if lock['commit'] != UPSTREAM_COMMIT: raise RuntimeError('INPUT_MISMATCH: OnlyTerp commit lock')
        for relative, expected in lock['runtime_files'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError('INPUT_MISMATCH: OnlyTerp source bytes changed: ' + relative)
        name = '_baseline_matrix_onlyterp_primary_mse'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'third_party/turboquant_onlyterp/src/cache.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        _SOURCE = module
    return _SOURCE


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in (1,2,3,4,8): raise ValueError('Unsupported categorical width')
    if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= (1 << bits)):
        raise ValueError('Categorical value exceeds declared width')
    shifts = torch.arange(bits, device=indices.device, dtype=torch.int64)
    expanded = ((indices.long().unsqueeze(-1) >> shifts) & 1).flatten(-2)
    padded = (8 - expanded.shape[-1] % 8) % 8
    if padded: expanded = F.pad(expanded, (0,padded))
    byte_shifts = torch.arange(8, device=indices.device, dtype=torch.int64)
    return ((expanded.reshape(*expanded.shape[:-1], -1, 8) << byte_shifts).sum(-1)).to(torch.uint8).contiguous()


def unpack_indices(packed: torch.Tensor, bits: int, count: int, dtype=torch.uint8) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.shape[-1] != (bits*count+7)//8:
        raise ValueError('Packed categorical shape/dtype mismatch')
    byte_shifts = torch.arange(8,device=packed.device,dtype=torch.int64)
    expanded = ((packed.long().unsqueeze(-1) >> byte_shifts)&1).flatten(-2)[...,:bits*count]
    shifts=torch.arange(bits,device=packed.device,dtype=torch.int64)
    return ((expanded.reshape(*expanded.shape[:-1],count,bits)<<shifts).sum(-1)).to(dtype)


@lru_cache(maxsize=16)
def _cpu_template(bits: int, outlier_bits: int, mixed_precision: bool):
    return _source().TurboQuantConfig(d=128,b_mse=bits,b_outlier=outlier_bits,
        mixed_precision=mixed_precision,n_outlier=32,use_online_codebook=False,
        rotation_mode='hadamard',device=torch.device('cpu'))


@lru_cache(maxsize=16)
def _cpu_outlier_codebook(bits: int):
    return _source().compute_lloyd_max_codebook(32,bits,device=torch.device('cpu'))


@lru_cache(maxsize=64)
def _device_codebooks(bits: int, outlier_bits: int, mixed_precision: bool, device_name: str):
    template=_cpu_template(bits,outlier_bits,mixed_precision)
    regular_dim=96 if mixed_precision else 128
    regular=replace(template.codebook,d=regular_dim)
    outlier=replace(_cpu_outlier_codebook(outlier_bits),d=32) if mixed_precision else None
    def move(book):
        return replace(book,centroids=book.centroids.to(device_name),boundaries=book.boundaries.to(device_name)) if book else None
    return regular,outlier,move(regular),move(outlier)


def _head_source_states(k: torch.Tensor,v: torch.Tensor,bits: int,outlier_bits: int,mixed_precision: bool,layer_idx: int,head_idx: int):
    source=_source(); template=_cpu_template(bits,outlier_bits,mixed_precision)
    if mixed_precision:
        # Same source functions and seed rule as get_mixed_config; only the
        # data-oblivious Lloyd-Max arrays are cached, never channel partitions.
        oi,ri=source.detect_outlier_channels(k,32)
        rr=template.make_subset_rotation(layer_idx,head_idx,'regular',ri.numel())
        ro=template.make_subset_rotation(layer_idx,head_idx,'outlier',oi.numel())
        rc,oc,_,_=_device_codebooks(bits,outlier_bits,True,str(k.device))
        mixed=source.MixedPrecisionConfig(n_outlier=32,b_regular=bits,b_outlier=outlier_bits,
            outlier_indices=oi,regular_indices=ri,codebook_regular=rc,codebook_outlier=oc,
            rotation_regular=rr,rotation_outlier=ro)
        rotation=rr  # unused uniform fallback; all mixed fields above present
    else:
        mixed=None; rotation=template.make_rotation(layer_idx,head_idx)
    return (source.turboquant_mse_encode(k,template.codebook,rotation,mixed=mixed),
            source.turboquant_mse_encode(v,template.codebook,rotation,mixed=mixed))


def _storage(tensors: dict[str,torch.Tensor]):
    seen=set();rows={};total=0
    for name,tensor in tensors.items():
        data=tensor.untyped_storage();key=(str(tensor.device),data.data_ptr(),data.nbytes())
        charge=0 if key in seen else data.nbytes();seen.add(key);total+=charge
        rows[name]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),
                    'storage_bytes':data.nbytes(),'charged_bytes':charge}
    return {'total_bytes':total,'tensors':rows}


@dataclass
class OnlyTerpMSEOwner:
    k_code: torch.Tensor
    v_code: torch.Tensor
    k_norms: torch.Tensor
    v_norms: torch.Tensor
    channel_maps: torch.Tensor
    rotation_sign_bits: torch.Tensor
    positions: torch.Tensor
    shape: tuple[int,...]
    bits_regular: int
    bits_outlier: int
    mixed_precision: bool
    layer_idx: int
    source_dtype: torch.dtype
    cpu_regular_codebook: Any
    cpu_outlier_codebook: Any
    regular_codebook: Any
    outlier_codebook: Any

    def named_tensors(self,include_constants=True):
        result={name:getattr(self,name) for name in ('k_code','v_code','k_norms','v_norms','channel_maps','positions')}
        if include_constants:
            result['encoder_constants.rotation_sign_bits']=self.rotation_sign_bits
            for prefix,book in [('gpu_regular',self.regular_codebook),('gpu_outlier',self.outlier_codebook),
                                ('cpu_source_regular',self.cpu_regular_codebook),('cpu_source_outlier',self.cpu_outlier_codebook)]:
                if book is not None:
                    result[f'encoder_constants.{prefix}.centroids']=book.centroids
                    result[f'encoder_constants.{prefix}.boundaries']=book.boundaries
        return result

    owner_tensors=named_tensors

    @property
    def regular_width(self): return (128*self.bits_regular+7)//8
    @property
    def outlier_width(self): return (32*self.bits_outlier+7)//8 if self.mixed_precision else 0

    def _rotation(self,sign_bits,dim):
        source=_source();rotation=object.__new__(source.RandomHadamardRotation)
        rotation.d=dim;rotation.seed=None;rotation.sqrt_d=math.sqrt(dim)
        rotation.signs=unpack_indices(sign_bits,1,dim,dtype=torch.float32)*2-1
        return rotation

    def _source_state(self,kind,head):
        source=_source();code=getattr(self,kind+'_code')[0,head];norms=getattr(self,kind+'_norms')[0,head]
        rr=self._rotation(self.rotation_sign_bits[head,:16],128)
        ri=unpack_indices(code[...,:self.regular_width],self.bits_regular,128)
        if not self.mixed_precision:
            return source.PolarQuantCompressed(norm=norms[:,0],indices=ri,
                codebook=self.regular_codebook,rotation=rr,original_dim=128)
        ro=self._rotation(self.rotation_sign_bits[head,16:],32)
        oi=unpack_indices(code[...,self.regular_width:],self.bits_outlier,32)
        return source.PolarQuantCompressed(original_dim=128,regular_norm=norms[:,0],outlier_norm=norms[:,1],
            regular_indices=self.channel_maps[head,:96].long(),outlier_indices=self.channel_maps[head,96:].long(),
            regular_quantized_indices=ri,outlier_quantized_indices=oi,
            codebook_regular=self.regular_codebook,codebook_outlier=self.outlier_codebook,
            rotation_regular=rr,rotation_outlier=ro)

    def materialize(self,dtype=None):
        # Rebuild only transient source states; no decoded K/V is stored in self.
        source=_source();outputs=[]
        for kind in ('k','v'):
            heads=[source.turboquant_mse_decode(self._source_state(kind,h)) for h in range(self.shape[1])]
            value=torch.stack(heads,dim=0).unsqueeze(0)
            outputs.append(value if dtype is None else value.to(dtype))
        return tuple(outputs)

    def attend(self,query,*,exact_keys=None,exact_values=None,exact_positions=None,query_positions=None,
               scale=None,backend=READER):
        if backend!=READER: raise ValueError('OnlyTerp MSE requires explicit FP32 materialize reference backend')
        b,hq,nq,d=query.shape;hkv=self.shape[1]
        if b!=1 or d!=128 or hq%hkv: raise ValueError('Unsupported batch/head geometry')
        k,v=self.materialize();q=query.float().reshape(b,hkv,hq//hkv,nq,d)
        logits=(q @ k.unsqueeze(2).transpose(-1,-2))*(d**-.5 if scale is None else scale)
        pos=self.positions.reshape(1,1,-1).expand(b,hkv,-1)
        if exact_keys is not None:
            if exact_values is None or exact_positions is None: raise ValueError('Exact text needs values and original positions')
            if exact_keys.shape!=exact_values.shape or exact_keys.shape[:2]!=(b,hkv): raise ValueError('Exact text geometry mismatch')
            logits=torch.cat((logits,(q @ exact_keys.float().unsqueeze(2).transpose(-1,-2))*(d**-.5 if scale is None else scale)),dim=-1)
            ep=exact_positions.to(query.device)
            if ep.ndim==1:ep=ep.reshape(1,1,-1).expand(b,hkv,-1)
            pos=torch.cat((pos,ep),dim=-1)
        if query_positions is not None:
            qp=query_positions.to(query.device)
            if qp.ndim==1:qp=qp[None,:].expand(b,-1)
            visible=pos[:,:,None,None,:]<=qp[:,None,None,:,None]
            logits=logits.masked_fill(~visible,-torch.inf)
        probabilities=logits.softmax(-1)
        if not torch.isfinite(probabilities).all():raise RuntimeError('Nonfinite OnlyTerp MSE joint softmax')
        output=probabilities[...,:self.shape[-2]] @ v.unsqueeze(2)
        if exact_keys is not None:output+=probabilities[...,self.shape[-2]:] @ exact_values.float().unsqueeze(2)
        return output.reshape(b,hq,nq,d).to(query.dtype)

    def owner_report(self):
        named=self.named_tensors();report=_storage(named);scalars=2*math.prod(self.shape)
        constants=_storage({k:v for k,v in named.items() if k.startswith('encoder_constants.')})['total_bytes']
        metadata=_storage({k:v for k,v in named.items() if k in ('k_norms','v_norms','channel_maps')})['total_bytes']
        return {'method':'TurboQuant-MSE_OnlyTerp_primary_visual_lossless_packed_reference','qjl_enabled':False,
            'method_paper':PAPER,'upstream_repo':UPSTREAM_REPO,'upstream_commit':UPSTREAM_COMMIT,
            'faithfulness_status':'UNMODIFIED_PRIMARY_MSE_ALIASES_LOSSLESS_CATEGORICAL_STORAGE_PORT',
            'reader':READER,'bits_nominal_MSE_payload_actual_padded_dimensions':(128*self.bits_regular+(32*self.bits_outlier if self.mixed_precision else 0))/128,
            'source_metadata_dtypes':{'norm':'torch.float16 (upstream)','codebook':'torch.float32 (upstream)',
                                      'map':'int64 values losslessly stored as uint8','rotation_sign':'float32 +/-1 values losslessly stored as 1bit'},
            'stored_visual_KV_bytes':report['total_bytes'],'actual_effective_visual_bits':8*report['total_bytes']/scalars,
            'full_kv_byte_denominator':2*scalars,'positions_bytes':self.positions.untyped_storage().nbytes(),
            'stored_visual_payload_bytes':self.k_code.untyped_storage().nbytes()+self.v_code.untyped_storage().nbytes(),
            'coder_metadata_bytes':metadata,'encoder_constants_bytes':constants,
            'temporary_full_FP32_visual_KV_bytes':4*scalars,
            'temporary_bit_expansion_and_exact_text_FP32_additional':True,
            'source_index_dimensions':{'regular':128,'outlier':32 if self.mixed_precision else 0},
            'original_group_dimensions':{'regular':96 if self.mixed_precision else 128,'outlier':32 if self.mixed_precision else 0},
            'all_required_GPU_and_shared_CPU_codebook_constants_named':True,'storage':report}

    byte_report=owner_report


class OnlyTerpMSEAdapter:
    def __init__(self,*,bits_regular=1,bits_outlier=None,mixed_precision=True,reader_backend=READER):
        if bits_regular not in (1,2,3):raise ValueError('Audited source MSE widths are 1,2,3')
        bits_outlier=bits_regular+1 if bits_outlier is None else bits_outlier
        if mixed_precision and bits_outlier!=bits_regular+1:raise ValueError('Use the declared source +1 outlier allocation')
        if reader_backend!=READER:raise ValueError('No silent experimental-kernel fallback')
        self.bits_regular,self.bits_outlier,self.mixed_precision=bits_regular,bits_outlier,bool(mixed_precision)
        self.reader_backend=reader_backend

    @torch.no_grad()
    def encode(self,k,v,*,positions=None,layer_idx=0,**kwargs):
        if k.ndim!=4 or k.shape!=v.shape or k.shape[0]!=1 or k.shape[-1]!=128 or k.shape[-2]==0:
            raise ValueError('OnlyTerp MSE supports batch-one matching [B,Hkv,N,128] tensors')
        if k.device!=v.device or k.dtype!=v.dtype or k.dtype not in (torch.float16,torch.float32):
            raise ValueError('K/V dtype/device mismatch')
        source=_source();n=k.shape[-2];hkv=k.shape[1]
        if positions is None:positions=torch.arange(n,device=k.device,dtype=torch.int64)
        positions=torch.as_tensor(positions,device=k.device,dtype=torch.int64).detach()
        if positions.ndim!=1 or positions.numel()!=n or positions.unique().numel()!=n:raise ValueError('Original visual positions must be unique 1D')
        if positions.min()<0:raise ValueError('Negative original visual position')
        if not positions.is_contiguous() or positions.storage_offset()!=0 or positions.untyped_storage().nbytes()!=positions.numel()*8:
            positions=positions.contiguous().clone()
        k_codes=[];v_codes=[];kn=[];vn=[];maps=[];signs=[]
        for head in range(hkv):
            ks,vs=_head_source_states(k[0,head],v[0,head],self.bits_regular,self.bits_outlier,self.mixed_precision,int(layer_idx),head)
            head_codes=[];head_norms=[]
            for state in (ks,vs):
                if self.mixed_precision:
                    codes=torch.cat((pack_indices(state.regular_quantized_indices,self.bits_regular),
                                     pack_indices(state.outlier_quantized_indices,self.bits_outlier)),dim=-1)
                    norms=torch.stack((state.regular_norm,state.outlier_norm),dim=-1)
                else:
                    codes=pack_indices(state.indices,self.bits_regular);norms=state.norm[:,None]
                head_codes.append(codes);head_norms.append(norms)
            k_codes.append(head_codes[0]);v_codes.append(head_codes[1]);kn.append(head_norms[0]);vn.append(head_norms[1])
            if self.mixed_precision:
                channel_map=torch.cat((ks.regular_indices,ks.outlier_indices)).to(device=k.device,dtype=torch.uint8)
                rotation_bits=torch.cat((pack_indices(((ks.rotation_regular.signs+1)/2).long(),1),
                                         pack_indices(((ks.rotation_outlier.signs+1)/2).long(),1))).to(k.device)
            else:
                channel_map=torch.empty(0,device=k.device,dtype=torch.uint8)
                rotation_bits=pack_indices(((ks.rotation.signs+1)/2).long(),1).to(k.device)
            maps.append(channel_map);signs.append(rotation_bits)
        cpu_r,cpu_o,gpu_r,gpu_o=_device_codebooks(self.bits_regular,self.bits_outlier,self.mixed_precision,str(k.device))
        owner=OnlyTerpMSEOwner(torch.stack(k_codes).unsqueeze(0),torch.stack(v_codes).unsqueeze(0),
            torch.stack(kn).unsqueeze(0),torch.stack(vn).unsqueeze(0),torch.stack(maps),torch.stack(signs),positions,
            tuple(k.shape),self.bits_regular,self.bits_outlier,self.mixed_precision,int(layer_idx),k.dtype,cpu_r,cpu_o,gpu_r,gpu_o)
        if owner.k_norms.dtype!=torch.float16 or owner.v_norms.dtype!=torch.float16:raise RuntimeError('Source norm dtype changed')
        return owner

    def manifest(self):
        return {'method':'TurboQuant-MSE_OnlyTerp_source_variant_QJL0','adapter':'methods.turboquant_onlyterp_mse_adapter.OnlyTerpMSEAdapter',
            'upstream_repo':UPSTREAM_REPO,'upstream_commit':UPSTREAM_COMMIT,'paper':PAPER,'qjl_enabled':False,
            'algorithm':'primary turboquant_mse_encode/decode; K-derived variance partition shared by K/V',
            'bits_regular':self.bits_regular,'bits_outlier':self.bits_outlier,'mixed_precision':self.mixed_precision,
            'original_dim':128,'original_regular_channels':96 if self.mixed_precision else 128,
            'stored_regular_coordinates':128,'stored_outlier_coordinates':32 if self.mixed_precision else 0,
            'source_construction_device':'CPU, then exact symbol upload (source default device is CPU)',
            'source_vector_arithmetic':'FP32','norm_dtype':'FP16 as original source','codebook_dtype':'FP32 as original source',
            'storage_changes':'lossless categorical packing only; codebook boundaries retained and charged',
            'reader_backend':self.reader_backend,'reference_reader_materializes_full_temporary_FP32_KV':True,
            'recent_visual_buffer_tokens':0,'experimental_Triton_and_buffer_paths_used':False,
            'online_codebook':False,'training_or_reference_answers_used':False,'claim_paper_prod_or_paper_2p5_table':False}
