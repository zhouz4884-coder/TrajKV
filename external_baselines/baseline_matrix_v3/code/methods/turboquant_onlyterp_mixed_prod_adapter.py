"""OnlyTerp primary mixed-MSE + one Gaussian QJL, visual-scope source port.

The source cache encodes QJL for both K and V, but its default attention reads
K=prod and V=MSE. Both V sign/norm payloads remain stored and charged here.
Full-prod V is an explicitly separate public decode-interface variant.
No experimental Rademacher kernel or double-QJL buffer manager is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import torch

from .turboquant_onlyterp_mse_adapter import (OnlyTerpMSEAdapter,_cpu_template,_source,_storage,
                                            pack_indices,unpack_indices,READER as MSE_READER)

ROOT=Path(__file__).resolve().parents[2]
SOURCE_LOCK='configs/methods/onlyterp_mixed_prod_source_lock_v1.json'
READER='torch_source_ip_fp32_reference'
_VERIFIED=False


def _guard():
    global _VERIFIED
    if not _VERIFIED:
        lock=json.loads((ROOT/SOURCE_LOCK).read_text(encoding='utf-8'))
        if lock['commit']!='1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306':raise RuntimeError('OnlyTerp mixed-prod source revision mismatch')
        for name,digest in lock['runtime_files'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Mixed-prod runtime source changed: '+name)
        _VERIFIED=True


@dataclass
class OnlyTerpMixedProdOwner:
    mse_owner:object
    k_qjl_signs:torch.Tensor
    v_qjl_signs:torch.Tensor
    k_qjl_rnorm:torch.Tensor
    v_qjl_rnorm:torch.Tensor
    gaussian_matrices:torch.Tensor
    value_read_mode:str='source_cache_mse'

    @property
    def positions(self):return self.mse_owner.positions
    @property
    def shape(self):return self.mse_owner.shape

    def named_tensors(self,include_constants=True):
        result=dict(self.mse_owner.named_tensors(include_constants))
        # k/v_packed identify additional Gaussian-QJL sign-bit payloads in the
        # shared owner's accounting. The MSE codes remain k/v_code.
        result.update(k_packed=self.k_qjl_signs,v_packed=self.v_qjl_signs,
                      k_qjl_rnorm=self.k_qjl_rnorm,v_qjl_rnorm=self.v_qjl_rnorm)
        if include_constants:result['encoder_constants.gaussian_qjl_matrices']=self.gaussian_matrices
        return result

    owner_tensors=named_tensors

    def source_state(self,kind,head):
        source=_source();pq=self.mse_owner._source_state(kind,head)
        signs=unpack_indices(getattr(self,kind+'_qjl_signs')[0,head],1,128,dtype=torch.int64)
        qjl=source.QJLCompressed(signs=signs,r_norm=getattr(self,kind+'_qjl_rnorm')[0,head],S=self.gaussian_matrices[head])
        return source.TurboQuantCompressed(pq=pq,qjl=qjl)

    def materialize(self,dtype=None):
        """Source public decode, with the chosen explicit V read interface."""
        source=_source();values=[]
        for kind in ('k','v'):
            heads=[]
            for head in range(self.shape[1]):
                state=self.source_state(kind,head)
                if kind=='v' and self.value_read_mode=='source_cache_mse':
                    value=source.turboquant_mse_decode(state.pq)
                else:value=source.turboquant_decode_single(state)
                heads.append(value)
            value=torch.stack(heads)[None]
            values.append(value if dtype is None else value.to(dtype))
        return tuple(values)

    def read_visual_logits(self,query,scale=None):
        """Primary cache's separate MSE score plus exactly one QJL score."""
        b,hq,nq,d=query.shape;hkv=self.shape[1]
        if b!=1 or d!=128 or hq%hkv:raise ValueError('Batch-one integral GQA with D128 required')
        k,_=self.mse_owner.materialize();q=query.float().reshape(b,hkv,hq//hkv,nq,d)
        factor=d**-.5 if scale is None else float(scale)
        mse_scores=(q @ k.unsqueeze(2).transpose(-1,-2))*factor
        projection=q @ self.gaussian_matrices[None,:,None].transpose(-1,-2)
        signs=unpack_indices(self.k_qjl_signs,1,128,dtype=torch.float32)*2-1
        correction=(projection @ signs.unsqueeze(2).transpose(-1,-2))*(math.sqrt(math.pi/2)/d)
        correction=correction*self.k_qjl_rnorm[:,:,None,None,:]*factor
        return (mse_scores+correction).reshape(b,hq,nq,-1)

    def attend(self,query,*,exact_keys=None,exact_values=None,exact_positions=None,query_positions=None,
               scale=None,backend=READER):
        if backend!=READER:raise ValueError('Declare the explicit source IP FP32 reference reader')
        b,hq,nq,d=query.shape;hkv=self.shape[1]
        if hq%hkv:raise ValueError('Nonintegral GQA')
        # This reference intentionally materializes source MSE FP32 K/V and
        # dense sign arrays. It is not an optimized packed-attention kernel.
        _,v=self.mse_owner.materialize()
        if self.value_read_mode=='full_prod_decode':v=self.materialize()[1]
        logits=self.read_visual_logits(query,scale);factor=d**-.5 if scale is None else float(scale)
        positions=self.positions.reshape(1,1,-1).expand(b,hkv,-1)
        q=query.float().reshape(b,hkv,hq//hkv,nq,d)
        if exact_keys is not None:
            if exact_values is None or exact_positions is None:raise ValueError('Exact text requires values and original positions')
            scores=(q @ exact_keys.float().unsqueeze(2).transpose(-1,-2))*factor
            logits=torch.cat((logits,scores.reshape(b,hq,nq,-1)),dim=-1)
            ep=exact_positions.to(query.device)
            if ep.ndim==1:ep=ep[None,None,:].expand(b,hkv,-1)
            positions=torch.cat((positions,ep),dim=-1)
        if query_positions is not None:
            qp=query_positions.to(query.device)
            if qp.ndim==1:qp=qp[None,:].expand(b,-1)
            mask=positions.repeat_interleave(hq//hkv,dim=1)[:,:,None,:]<=qp[:,None,:,None]
            logits=logits.masked_fill(~mask,-torch.inf)
        p=logits.softmax(-1)
        if not torch.isfinite(p).all():raise RuntimeError('Nonfinite single-QJL joint softmax')
        pg=p[...,:self.shape[-2]].reshape(b,hkv,hq//hkv,nq,-1)
        output=pg @ v.float().unsqueeze(2)
        if exact_keys is not None:
            pe=p[...,self.shape[-2]:].reshape(b,hkv,hq//hkv,nq,-1)
            output+=pe @ exact_values.float().unsqueeze(2)
        return output.reshape(b,hq,nq,d).to(query.dtype)

    def owner_report(self):
        named=self.named_tensors();report=_storage(named);scalars=2*math.prod(self.shape)
        payload=sum(t.untyped_storage().nbytes() for n,t in named.items() if n in ('k_code','v_code','k_packed','v_packed'))
        constants=_storage({n:t for n,t in named.items() if n.startswith('encoder_constants.')})['total_bytes']
        return {'method':'OnlyTerp_mixed_prod_single_GaussianQJL_visual_reference','method_paper':'https://arxiv.org/abs/2504.19874v1',
            'upstream_repo':'https://github.com/OnlyTerp/turboquant','upstream_commit':'1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306',
            'faithfulness_status':'PRIMARY_MIXED_MSE_PLUS_ONE_GLOBAL_GAUSSIAN_QJL_SOURCE_PORT',
            'K_read':'MSE score plus exactly one Gaussian-QJL correction','V_read':self.value_read_mode,
            'V_QJL_encoding_stored_and_charged_even_if_source_default_read_ignores_it':True,
            'bits_regular_MSE':self.mse_owner.bits_regular,'bits_outlier_MSE':self.mse_owner.bits_outlier,
            'actual_padded_MSE_payload_bits':(128*self.mse_owner.bits_regular+32*self.mse_owner.bits_outlier)/128,
            'additional_QJL_sign_payload_bits_per_KV_scalar':1.,'norm_dtypes':{'MSE_group_norm':'float16 source','QJL_residual_norm':'float32 source'},
            'projection_matrix_dtype':'float32 source Gaussian','persistent_projection_matrix_device':str(self.gaussian_matrices.device),
            'projection_matrix_bytes':self.gaussian_matrices.untyped_storage().nbytes(),'CPU_Gaussian_generation_scratch_retained':False,
            'encoder_constants_bytes':constants,'stored_visual_payload_bytes':payload,
            'coder_metadata_bytes':report['total_bytes']-payload-constants-self.positions.untyped_storage().nbytes(),
            'positions_bytes':self.positions.untyped_storage().nbytes(),'stored_visual_KV_bytes':report['total_bytes'],
            'full_kv_byte_denominator':2*scalars,'actual_effective_visual_bits':8*report['total_bytes']/scalars,
            'reader_backend':READER,'temporary_full_FP32_MSE_visual_KV_bytes':4*scalars,
            'additional_temporary_dense_signs_query_projection_bit_expansion':True,
            'reader_repeats_some_reference_materialization':True,'not_paper_optimized_kernel_or_paper_2p5_table':True,'storage':report}

    byte_report=owner_report


class OnlyTerpMixedProdAdapter:
    def __init__(self,*,bits_regular=3,bits_outlier=None,value_read_mode='source_cache_mse',reader_backend=READER):
        bits_outlier=bits_regular+1 if bits_outlier is None else bits_outlier
        if bits_regular not in (2,3) or bits_outlier!=bits_regular+1:raise ValueError('Use declared source MSE2/3 or source-default3/4 groups')
        if value_read_mode not in ('source_cache_mse','full_prod_decode'):raise ValueError('Declare the V read interface')
        if reader_backend!=READER:raise ValueError('No silent Rademacher/experimental-kernel fallback')
        _guard();self.bits_regular=bits_regular;self.bits_outlier=bits_outlier
        self.value_read_mode=value_read_mode;self.reader_backend=reader_backend

    @torch.no_grad()
    def encode(self,k,v,*,positions=None,layer_idx=0,**kwargs):
        base=OnlyTerpMSEAdapter(bits_regular=self.bits_regular,bits_outlier=self.bits_outlier,
            mixed_precision=True,reader_backend=MSE_READER).encode(k,v,positions=positions,layer_idx=layer_idx)
        kh,vh=base.materialize();source=_source();template=_cpu_template(self.bits_regular,self.bits_outlier,True)
        signs_k=[];signs_v=[];norms_k=[];norms_v=[];matrices=[]
        for head in range(k.shape[1]):
            # make_qjl_matrix is the source CPU Gaussian generator and seed rule.
            # Only its exact device copy is persistent; source CPU scratch dies.
            matrix=template.make_qjl_matrix(int(layer_idx),head).to(device=k.device,dtype=torch.float32)
            kc=source.qjl_encode(k[0,head].detach().float()-kh[0,head],matrix)
            vc=source.qjl_encode(v[0,head].detach().float()-vh[0,head],matrix)
            if kc.r_norm.dtype!=torch.float32 or vc.r_norm.dtype!=torch.float32:raise RuntimeError('Source QJL residual norm dtype changed')
            signs_k.append(pack_indices(kc.signs,1));signs_v.append(pack_indices(vc.signs,1))
            norms_k.append(kc.r_norm);norms_v.append(vc.r_norm);matrices.append(matrix)
        return OnlyTerpMixedProdOwner(base,torch.stack(signs_k)[None],torch.stack(signs_v)[None],
            torch.stack(norms_k)[None],torch.stack(norms_v)[None],torch.stack(matrices),self.value_read_mode)

    def manifest(self):
        return {'method':'OnlyTerp_mixed_prod_single_GaussianQJL_visual_source_variant','source_lock':SOURCE_LOCK,
            'bits_regular_MSE':self.bits_regular,'bits_outlier_MSE':self.bits_outlier,'outlier_channels':32,'regular_channels':96,
            'regular_padded_dimension':128,'outlier_rotation_dimension':32,'Gaussian_QJL_dimension':128,
            'K_QJL_corrections':1,'V_encoding':'MSE plus Gaussian QJL as source store_batch','V_read':self.value_read_mode,
            'source_value_default':'source_cache_mse','full_prod_V_is_separate_decode_interface_variant':True,
            'MSE_norm_dtype':'FP16 source','QJL_residual_norm_dtype':'FP32 source','projection_dtype':'FP32 source',
            'categorical_changes':'lossless packing only','Gaussian_matrices_all_persistent_copies_named_and_charged':True,
            'reader_backend':READER,'experimental_kernels_or_buffer_manager_used':False,'training_or_GT_used':False,
            'scope':'immutable visual prefix + exact FP16 text, not full rolling source cache or paper 2.5/3.5 table reproduction'}
