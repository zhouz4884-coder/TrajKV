"""Round10 factory v2: absent residual layers preserved, frozen v1 math reused.

The QA0 window kernel remains exactly in v1. Only whole-model factory presence
checks change here; old/staged v1 bytes and GPU evidence remain immutable.
"""
from __future__ import annotations
from copy import copy
import importlib
from pathlib import Path
import time
import torch
import round10_value_bank_v1 as numerical

old=numerical.old
bridge=numerical.bridge
MODES=numerical.MODES
WINDOWS=numerical.WINDOWS
THETA_SCHEMA=numerical.THETA_SCHEMA
NUMERICAL_PINS={**numerical.NUMERICAL_PINS,"round10_value_bank_v1.py":"5c9880d9e7facdc1623473dd74759aee9064bd5ff313f1581390fe0ccd3aae1c"}
require=numerical.require
sha=numerical.sha
tensor_sha=numerical.tensor_sha
normalize_window=numerical.normalize_window
load_theta=numerical.load_theta
_theta=numerical._theta
_fields=numerical._fields
feedback_assign_window=numerical.feedback_assign_window

def check_numerical_pins():
    numerical.check_numerical_pins()
    require(sha(numerical.__file__)==NUMERICAL_PINS["round10_value_bank_v1.py"],"frozen staged v1 changed")

def check_residual_presence(previous,state,*,same_objects):
    for name in ("depth_residual_code","depth_residual_alpha"):
        require(hasattr(previous,name)==hasattr(state,name),"residual field presence changed: "+name)
        if hasattr(previous,name) and same_objects:
            require(getattr(previous,name) is getattr(state,name),"frozen old residual replaced: "+name)

@torch.no_grad()
def build_round10_value_owners(native_prefix_cache, visual_positions, variant, *, mode,
        theta_path=None, theta_sha256=None, theta_override=None, calibration_only=False,
        window=None, tile_tokens=256, legacy_a2, legacy_experiments, legacy_scripts, kivi_source,
        expected_asset_sha256=None, expected_legacy_source_sha256=None):
    check_numerical_pins()
    require(mode in MODES, "unknown Round10 mode")
    ls = mode in ("LS-FZ", "LS-RE")
    require((theta_override is None or calibration_only) and not(theta_override is not None and theta_path is not None), "in-memory theta only for explicit calibration")
    require(theta_override is None or theta_sha256 is None, "in-memory theta must not claim a file SHA")
    require((ls and window is None) or (not ls and theta_path is None and theta_sha256 is None and theta_override is None), "LS/FB families must not be mixed")
    require(not ls or theta_path is not None or theta_override is not None, "LS needs its exact absolute theta")
    if not ls:
        require(mode != "FB0" or window is None or window == 1, "FB0 cannot silently ignore a different window")
        window = 1 if mode == "FB0" else normalize_window(window)
    sources = [layer.values.index_select(2,visual_positions).detach().contiguous() for layer in native_prefix_cache.layers]
    owners, metadata = bridge.build_r7_owners(native_prefix_cache,visual_positions,variant,
        legacy_a2=legacy_a2,legacy_experiments=legacy_experiments,legacy_scripts=legacy_scripts,kivi_source=kivi_source,
        expected_asset_sha256=expected_asset_sha256,expected_legacy_source_sha256=expected_legacy_source_sha256)
    require(len(owners)==28 and metadata["selected_stage_count"]==43 and
            sum(x[0]=="k" for x in metadata["selected_stages"])==23 and sum(x[0]=="v" for x in metadata["selected_stages"])==20, "original K23/V20 only")
    before_bytes = old._unique(owners)
    if ls:
        if theta_override is not None:
            theta = _theta(theta_override,device=visual_positions.device)
            theta_info = {"origin":"calibration_only_in_memory","schema":THETA_SCHEMA,"parameter_count":224}
        else:
            theta,theta_info = load_theta(theta_path,theta_sha256,device=visual_positions.device)
        theta_info["tensor_sha256"]={name:tensor_sha(value) for name,value in theta.items()}
    else:
        theta = None;theta_info={"origin":"original_V0_direct_physical_fields_no_second_fold"}
    stats=[]
    with bridge._legacy_paths(Path(legacy_a2).resolve(),Path(legacy_experiments).resolve(),Path(legacy_scripts).resolve()):
        quant=importlib.import_module("quantizers");fold=importlib.import_module("affine_rate_bank").fold
        for layer,(owner,source) in enumerate(zip(owners,sources)):
            started=time.perf_counter();previous=owner.state;state=copy(previous)
            item={"layer":layer,"before_V_field_sha256":_fields(previous)}
            if ls:
                raw=quant.binary_centroids(source.transpose(2,3).contiguous(),group_size=source.shape[-2])
                require(torch.equal(raw[0],previous.v_code),"original unfolded binary assignment differs")
                _,state.v_scale,state.v_min=old.apply_theta_to_fields(raw,theta,layer_idx=layer,fold_fn=fold)
                if mode=="LS-RE":
                    state,detail=old.reencode_value_residuals(state,source,metadata["selected_stages"],layer);item.update(detail)
            else:
                state,detail=feedback_assign_window(previous,source,window=window,tile_tokens=tile_tokens);item.update(detail)
            for name in ("direction_code","direction_gain","direction_mean","depth_residual_lookup","visual_pos"):
                require(getattr(state,name) is getattr(previous,name),"K/lookup/positions replaced: "+name)
            for slot,(axis,_,_,_) in enumerate(x for x in metadata["selected_stages"] if x[1]==layer):
                if axis=="k":
                    require(torch.equal(state.depth_residual_code[slot],previous.depth_residual_code[slot]) and torch.equal(state.depth_residual_alpha[slot],previous.depth_residual_alpha[slot]),"K residual changed")
            check_residual_presence(previous,state,same_objects=mode!="LS-RE")
            if not ls:
                require(state.v_scale is previous.v_scale and state.v_min is previous.v_min,"FB changed physical old V0")
            require(bool(torch.isfinite(state.v_scale).all() & torch.isfinite(state.v_min).all()),"nonfinite folded V")
            owner.state=state
            item.update(after_V_field_sha256=_fields(state),K_fields_unchanged=True,
                        host_seconds_including_synchronizing_diagnostics=time.perf_counter()-started)
            stats.append(item)
        if ls:
            # Both original f0 and the real new V224 encoding asset are retained.
            owners[0].shared.update({"round10_theta."+name:value for name,value in theta.items()})
    after_bytes=old._unique(owners);delta=896 if ls else 0
    require(after_bytes==before_bytes+delta,"all persistent constants must be charged; expected +896 LS / +0 FB")
    require(all(bridge._sha(path)==item["sha256"] for path,item in metadata["verified_legacy_source_pins"].items()),"legacy changed during transform")
    require(all(bridge._sha(item["path"])==item["sha256"] for item in metadata["fixed_assets"].values()),"base assets changed")
    forbidden={bridge._storage(t) for layer in native_prefix_cache.layers for t in(layer.keys,layer.values)}
    for owner in owners:
        for name,tensor in owner.named_tensors().items():
            require(not tensor.requires_grad and tensor.grad_fn is None,"autograd retained")
            require(bridge._storage(tensor) not in forbidden,"native prefix/source retained: "+name)
    metadata["round10_value_candidate"]={"schema":"trajkv.round10.value_candidate.v1","mode":mode,
        "theta":theta_info,"window":window,"tile_tokens":tile_tokens,"calibration_only":bool(calibration_only),
        "fixed_K":True,"original_K23_V20_coverage":True,"reader_unchanged":True,"old_f0_retained_and_charged":True,
        "new_theta_asset_bytes":delta,"before_unique_capacity_bytes":before_bytes,"after_unique_capacity_bytes":after_bytes,
        "actual_extra_persistent_bytes":after_bytes-before_bytes,"same_packed_payload_layout":True,
        "source_sha256":sha(__file__),"numerical_source_pins":NUMERICAL_PINS,"layer_stats":stats,
        "native_FP16_source_or_feedback_workspace_retained":False,"quality_or_speed_gain_claimed":False}
    metadata["returned_owners_unique_tensor_capacity_bytes"]=after_bytes
    return owners,metadata
