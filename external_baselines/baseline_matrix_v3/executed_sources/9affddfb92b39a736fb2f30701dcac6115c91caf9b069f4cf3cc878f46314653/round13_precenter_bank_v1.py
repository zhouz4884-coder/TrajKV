"""Position-conditioned K center, preserving centered-f0 base and raw RE target.

Prototype only. Frozen R is built first; its complete owner is the request cap.
Only K residual donors are removed; every R V field stays bit-identical.
Prefix RoPE frequencies are caller-supplied frozen state, never future query
state. Phase tables are transient encode/read scratch and are not persisted.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import torch
import r7_legacy_bridge_v1 as bridge
import round12_static_bank_v1 as previous

SCHEMA='trajkv.round13.precenter.prototype.v1'
RUNTIME_FILES=('round13_precenter_bank_v1.py','round13_precenter_attention_v1.py',
               'round13_precenter_technical_v1.py')
EXTRA_BYTES=28*4*128*2+64*4+4+4


def require(value,message):
    if not value:raise ValueError('ROUND13_PRECENTER: '+message)


def r_count(n):
    return min(224,(1670*int(n)+65408)//(18*int(n)))


def budget_plan(n, *, r_stages=None, cap_bytes=None):
    n=int(n);stages=r_count(n) if r_stages is None else int(r_stages)
    require(n>0 and n%32==0,'complete original visual words')
    cap=(3816+18*stages)*n+89737 if cap_bytes is None else int(cap_bytes)
    donors=math.ceil(EXTRA_BYTES/(18*n))
    return {'N':n,'original_R_stages':stages,'R_cap_bytes':cap,
        'K_donor_stages':donors,'selected_stages':stages-donors,
        'new_mu_bytes':28672,'frozen_frequency_bytes':256,
        'frozen_scaling_bytes':4,'new_descriptor_bytes':4,'extra_PC_bytes':EXTRA_BYTES,
        'donor_only_actual_bytes':cap-donors*18*n,
        'PC_expected_actual_bytes':cap-donors*18*n+EXTRA_BYTES,
        'PC_unused_cap_bytes':donors*18*n-EXTRA_BYTES,
        'donor_unused_cap_bytes':donors*18*n,
        'equal_actual_bytes_claimed':False}


def phase_tables(positions,inv_freq,attention_scaling,*,rotation_constant=False):
    """Match native Qwen2 Torch FP32 phase + Half c/s, including pair layout.

    This allocates transient tables on every call. It is a correctness-first
    prototype, explicitly not a fused low-overhead trigonometry claim.
    """
    require(positions.ndim==1 and inv_freq.shape==(64,),'serial position/frequency shape')
    if rotation_constant:
        shape=(1,1,positions.numel(),128)
        return (torch.ones(shape,device=positions.device,dtype=torch.float16),
                torch.zeros(shape,device=positions.device,dtype=torch.float16))
    freq=inv_freq[None,:,None].float()
    pos=positions[None,None,:].float()
    phase=(freq@pos).transpose(1,2)
    emb=torch.cat((phase,phase),-1)
    scale=attention_scaling.float() if isinstance(attention_scaling,torch.Tensor) else float(attention_scaling)
    return ((emb.cos()*scale).half().unsqueeze(1).contiguous(),
            (emb.sin()*scale).half().unsqueeze(1).contiguous())


def position_mean_bias(mu,a,g,cos,sin,*,rotation_constant=False):
    """The single deployed rounding definition shared by encode and reference.

    s = Half(Half(mu*cos)+Half(rotate_half(mu)*sin)).
    b = Half(a_float + g_float*(s_float-a_float)).
    The anchored affine expression makes s=a imply b=a exactly.
    """
    if rotation_constant:
        s=mu.expand(*mu.shape[:-2],cos.shape[-2],128)
    else:
        rotated=torch.cat((-mu[...,64:],mu[...,:64]),-1)
        s=((mu*cos).half()+(rotated*sin).half()).half()
    b=(a.float()+g.float()*(s.float()-a.float())).half()
    return s,b


def least_squares_mu(source,cos,sin):
    """Fit A_i mu to actual post-RoPE source using represented Half c/s."""
    require(source.dtype==torch.float16 and source.shape[-1]==128,'native post-K Half D128')
    x=source.float();c=cos[...,:64].float();s=sin[...,:64].float()
    den=(c.square()+s.square()).sum(-2,keepdim=True).clamp_min(1e-30)
    a=(c*x[...,:64]+s*x[...,64:]).sum(-2,keepdim=True)/den
    b=(-s*x[...,:64]+c*x[...,64:]).sum(-2,keepdim=True)/den
    return torch.cat((a,b),-1).half().contiguous()


def donor_subset(original,donors):
    require(all(int(x[3])==0 for x in original),'frozen R first-depth ranking')
    keys=[list(x) for x in original if x[0]=='k']
    require(len(keys)>=donors,'not enough K slots; never donate V')
    dropped=keys[-donors:] if donors else []
    selected=[list(x) for x in original if list(x) not in dropped]
    return selected,dropped


def reduce_residual_pool(state,original,selected,layer,lookup):
    old=previous._layer_entries(original,layer);new=previous._layer_entries(selected,layer)
    if not new:
        for name in ('depth_residual_code','depth_residual_alpha'):
            if hasattr(state,name):delattr(state,name)
        state.depth_residual_lookup=lookup[layer]
        return new
    slots=torch.tensor([old.index(x) for x in new],device=state.visual_pos.device,dtype=torch.long)
    rc,ra=state.depth_residual_code,state.depth_residual_alpha
    state.depth_residual_code=rc.index_select(0,slots).contiguous()
    state.depth_residual_alpha=ra.index_select(0,slots).contiguous()
    for slot,(axis,head,depth) in enumerate(new):
        lookup[layer,int(axis=='v'),head,depth]=slot
        if axis=='v':
            j=old.index((axis,head,depth))
            require(torch.equal(state.depth_residual_code[slot],rc[j]) and
                    torch.equal(state.depth_residual_alpha[slot],ra[j]),'V residual bit identity')
    state.depth_residual_lookup=lookup[layer]
    return new


class PrecenterOwner(previous.Round12Owner):
    def named_tensors(self,include_constants=True):
        result=super().named_tensors(include_constants)
        for name in ('round13_precenter_mu','round13_precenter_freq',
                     'round13_precenter_scaling','round13_precenter_gain'):
            value=getattr(self.state,name,None)
            if isinstance(value,torch.Tensor):result['encoder_constants.'+name+'_view']=value
        return result
    owner_tensors=named_tensors


@torch.no_grad()
def build_round13_precenter_owners(native_prefix_cache,visual_positions,variant,*,
        candidate,frozen_prefix_inv_freq,frozen_prefix_attention_scaling=1.0,
        legacy_a2,legacy_experiments,legacy_scripts,kivi_source,
        expected_asset_sha256=None,expected_legacy_source_sha256=None):
    started=time.perf_counter();mode=candidate.get('mode','PC')
    require(mode in ('PC','donor'),'PC or same-K-donor control required')
    technical=bool(candidate.get('technical_only',False))
    constant=bool(candidate.get('technical_rotation_constant',False))
    zero=bool(candidate.get('technical_mu_zero',False))
    require(technical or not (constant or zero),'technical center controls are not production methods')
    require(not (constant and zero),'choose one technical degeneration')
    n=int(visual_positions.numel());device=visual_positions.device
    require(isinstance(frozen_prefix_inv_freq,torch.Tensor) and
            frozen_prefix_inv_freq.dtype==torch.float32 and
            frozen_prefix_inv_freq.shape==(64,) and
            bool(torch.isfinite(frozen_prefix_inv_freq).all()),'actual frozen prefix 64-FP32 inv_freq required')
    require(math.isclose(float(frozen_prefix_attention_scaling),1.0,rel_tol=0,abs_tol=0),
            'this M1 prototype is restricted to native dynamic-RoPE attention_scaling1')
    kwargs=dict(legacy_a2=legacy_a2,legacy_experiments=legacy_experiments,
        legacy_scripts=legacy_scripts,kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,
        expected_legacy_source_sha256=expected_legacy_source_sha256)
    owners0,metadata=previous.build_round12_owners(native_prefix_cache,visual_positions,variant,
        candidate={'mode':'R'},**kwargs)
    original=deepcopy(metadata['selected_stages']);cap=previous.unique_capacity(owners0)
    plan=budget_plan(n,r_stages=len(original),cap_bytes=cap)
    selected,dropped=donor_subset(original,plan['K_donor_stages'])
    shared=owners0[0].shared;lookup=shared['global_residual_lookup'];lookup.fill_(-1)
    f0=json.loads(Path(variant['metadata_factor_path']).read_text(encoding='utf-8-sig'))
    require(f0['key_gain_mode']=='centered','preserve actual original centered-f0 profile')
    if mode=='PC':
        freq=frozen_prefix_inv_freq.detach().to(device=device).clone().contiguous()
        scale=torch.tensor([float(frozen_prefix_attention_scaling)],device=device,dtype=torch.float32)
        mu_pool=torch.empty((28,4,1,128),device=device,dtype=torch.float16)
        shared['round13_precenter_mu']=mu_pool
        shared['round13_precenter_freq']=freq
        shared['round13_precenter_scaling']=scale
        shared['round13_precenter_format']=torch.tensor([13,1,int(constant),int(zero)],device=device,dtype=torch.uint8)
        cos,sin=phase_tables(visual_positions,freq,scale,rotation_constant=constant)
    result=[];details=[]
    with bridge._legacy_paths(Path(legacy_a2),Path(legacy_experiments),Path(legacy_scripts)):
        encode=importlib.import_module('round7_key_codec').encode_key
        decode=importlib.import_module('round7_key_codec').decode_key
        sign=importlib.import_module('round6_depth_codec').encode_sign_depth
        wht=importlib.import_module('fast_wht').wht
        reader=importlib.import_module('round13_precenter_attention_v1').attention if mode=='PC' else owners0[0]._reader
        factors=shared['encoder_factor.key_head_factors']
        for layer,old in enumerate(owners0):
            state=old.state
            entries=reduce_residual_pool(state,original,selected,layer,lookup)
            detail={'layer':layer,'entries':entries,'V_fields_untouched':True}
            if mode=='PC':
                source=native_prefix_cache.layers[layer].keys.index_select(2,visual_positions).detach().contiguous()
                a=state.direction_mean
                # The constant-rotation control takes the already deployed
                # Half mean, avoiding a different parallel reduction as a
                # confound in this exact legacy-degeneration instrument.
                mu=a.detach().clone() if constant else least_squares_mu(source,cos,sin)
                if zero:mu.zero_()
                mu_pool[layer].copy_(mu[0])
                state.round13_precenter_mu=mu_pool[layer].unsqueeze(0)
                state.round13_precenter_freq=freq
                state.round13_precenter_scaling=scale
                state.round13_precenter_gain=factors[layer]
                state.round13_precenter_rotation_constant=constant
                g=factors[layer][None,:,None,None]
                s,b=position_mean_bias(mu,a,g,cos,sin,rotation_constant=constant)
                base_source=wht(source.float()-s.float())
                target=wht(source.float()-b.float())
                old_code,old_gain=state.direction_code,state.direction_gain
                code,gain=encode(base_source,shared['book'],amplitude='norm',assignment='l2',gain_factor=g)
                state.direction_code,state.direction_gain=code,gain
                base=decode(code,gain,shared['book'])
                detail.update(mu_energy=float(mu.float().square().sum()),
                    base_code_equal_original=bool(torch.equal(code,old_code)),
                    base_gain_bit_equal_original=bool(torch.equal(gain.view(torch.int16),old_gain.view(torch.int16))))
                for head in sorted({head for axis,head,_ in entries if axis=='k'}):
                    depths=[depth for axis,h,depth in entries if axis=='k' and h==head]
                    require(depths==list(range(len(depths))),'ordered K residual depths')
                    codes,alphas,_,distortion=sign(target[0,head],base[0,head],len(depths))
                    for depth in depths:
                        slot=entries.index(('k',head,depth))
                        state.depth_residual_code[slot].copy_(codes[depth])
                        state.depth_residual_alpha[slot].copy_(alphas[depth])
                    detail.setdefault('K_residual_distortion',[]).append({'head':head,'values':distortion})
                del source,mu,s,b,base_source,target,base,old_code,old_gain
            result.append(PrecenterOwner(layer,state,shared,reader,old.boundary))
            details.append(detail)
    owners0.clear()
    actual=previous.unique_capacity(result)
    expected=plan['PC_expected_actual_bytes'] if mode=='PC' else plan['donor_only_actual_bytes']
    require(actual==expected and actual<=cap,f'actual complete owner/cap mismatch: {actual} vs {expected}')
    native_ids={bridge._storage(t) for layer in native_prefix_cache.layers for t in (layer.keys,layer.values)}
    if mode=='PC':native_ids.add(bridge._storage(frozen_prefix_inv_freq))
    extra={'round13_precenter_mu','round13_precenter_freq','round13_precenter_scaling','round13_precenter_gain'}
    allowed=set(bridge._STATE_TENSORS)|extra
    for owner in result:
        require(all(not isinstance(t,torch.Tensor) or not t.numel() or name in allowed
                    for name,t in vars(owner.state).items()),'unreported persistent state tensor')
        for name,t in owner.named_tensors().items():
            require(bridge._storage(t) not in native_ids and not t.requires_grad and t.grad_fn is None,
                    'native source/frequency alias or autograd: '+name)
    metadata['reference_R']=metadata.pop('round12_static')
    metadata.update(selected_stages=selected,selected_stage_count=len(selected),
        returned_owners_unique_tensor_capacity_bytes=actual,reader_unchanged=mode=='donor',
        original_FP16_visual_storage_retained_by_owners=False)
    metadata['round13_precenter']={'schema':SCHEMA,'mode':mode,**plan,
        'actual_owner_bytes':actual,'unused_cap_bytes':cap-actual,
        'original_R_selected_stages':original,'K_donor_stages_removed':dropped,
        'V_selected_stages_preserved':[x for x in original if x[0]=='v'],
        'V_all_code_scale_min_and_logical_residuals_bit_preserved':True,
        'prefix_frequencies_copied_frozen_not_future_query':True,
        'freq_sha256':hashlib.sha256(frozen_prefix_inv_freq.detach().cpu().numpy().tobytes()).hexdigest(),
        'new_persistent_rope_and_mu_bytes':EXTRA_BYTES if mode=='PC' else 0,
        'f0_mode':'centered','base_source':'H(native_post_K-HalfRoPE(mu))',
        'bias_source':'Half(a_float+g_float*(s_float-a_float))',
        'K_residual_target':'H(native_post_K-deployed_bias), not H(K-s)',
        'phase_tables_persistent':False,'phase_tables_ephemeral_Half_bytes_per_attention_call':512*n if mode=='PC' else 0,
        'phase_table_FP32_intermediate_peak_not_measured_by_factory':True,
        'phase_table_generation':'Torch FP32 native-style phase/cos/sin per layer attention call; correctness-first prototype',
        'technical_only':technical,'technical_rotation_constant':constant,'technical_mu_zero':zero,
        'layer_stats':details,'encoding_seconds_including_R':time.perf_counter()-started,
        'CUDA_equivalence_established_by_factory':False,
        'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in RUNTIME_FILES}}
    return result,metadata
