"""Root-invoked synthetic B1 controls; no model/image/GT and no fitting.

--cpu-only runs small algebra/immutability controls. GPU acceptance is separate:
two original geometries Nv256/1792, L28/Hkv4/Hq28/D128 on FP16 SM86.
"""
from __future__ import annotations

import argparse
import ast
from copy import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import torch
import round9_value_bank_v1 as ops
import r7_bridge_gpu_check_v1 as reference
import r7_legacy_bridge_v1 as bridge

SCHEMA = "trajkv.round9.value_synthetic_control.v1"
NV = (256, 1792)
BRIDGE_SHA = "b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"
REFERENCE_SHA = "e37810a2568ebb21f10ea10488829be0758311985c5dbdeebdd20fa450c5631a"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
def require(condition, message):
    if not condition: raise ValueError(message)


def small_state(device="cpu"):
    torch.manual_seed(0)
    bits = torch.randint(0, 2, (1, 4, 128, 32), device=device)
    lookup = torch.full((2, 4, 2), -1, dtype=torch.int32, device=device)
    lookup[0, 0, 0] = 0; lookup[1, 1, 0] = 1; lookup[1, 1, 1] = 2
    return SimpleNamespace(v_code=ops._pack(bits),
        v_scale=(torch.rand(1,4,128,1,device=device)*.2+.05).half(),
        v_min=(torch.randn(1,4,128,1,device=device)*.05).half(),
        depth_residual_code=ops._pack(torch.randint(0,2,(3,32,128),device=device)),
        depth_residual_alpha=(torch.rand(3,32,device=device)*.02).half(), depth_residual_lookup=lookup)


def cpu_controls(a2):
    tests = []
    state = small_state()
    source = torch.randn(1,4,32,128).half()*.15
    snapshots = {key: value.clone() for key,value in vars(state).items()}
    dense = ops.decode_value_fields(state)
    # Independent scalar-field expansion and explicit old sequential stages.
    bits = ops._unpack(state.v_code,32)
    base = (bits.half()*state.v_scale+state.v_min).half().transpose(2,3).contiguous()
    for head in range(4):
        for stage in range(2):
            slot = int(state.depth_residual_lookup[1,head,stage])
            if slot >= 0:
                words = state.depth_residual_code[slot].long() & 0xffffffff
                sign = ((words[...,None] >> torch.arange(32)) & 1).reshape(32,128).float()*2-1
                base[:,head] = (base[:,head].float()+sign*state.depth_residual_alpha[slot,:,None].float()).half()
    require(torch.equal(dense,base), "CPU actual Half sequential decode")
    tests.append("actual_Half_base_and_depth2_decode_bitwise")
    f0, _ = ops.feedback_assign(state, source, feedback=0)
    a,b=(ops.decode_value_fields(state,base_bits_override=x) for x in (0,1))
    require(torch.equal(f0.v_code,ops._pack(((source.float()-b.float()).abs()<(source.float()-a.float()).abs()).transpose(2,3).contiguous())), "FB0 fixed level oracle")
    tests.append("FB0_actual_old_residual_levels_bitwise")
    tie = copy(state); tie.v_scale=torch.ones_like(state.v_scale);tie.v_min=torch.zeros_like(state.v_min)
    tie.depth_residual_lookup=torch.full_like(state.depth_residual_lookup,-1)
    t0,_=ops.feedback_assign(tie,torch.full_like(source,.5),feedback=0)
    require(not bool(t0.v_code.any()), "exact tie must choose0")
    tests.append("FB_tie0")
    zero,_=ops.feedback_assign(tie,torch.zeros_like(source),feedback=1,backend="torch")
    require(not bool(zero.v_code.any()), "zero source carry identity")
    tests.append("FB1_zero_input_no_carry_drift")
    f1,s1=ops.feedback_assign(state,source,feedback=1,backend="torch")
    deployed=ops.decode_value_fields(f1)
    terminal=(source.float()-deployed.float()).sum(2)
    # The scan accumulates FP32 in order; re-run independently in that order.
    carry=torch.zeros_like(source[:,:,0],dtype=torch.float32)
    oracle=[]
    for token in range(32):
        u=source[:,:,token].float()+carry
        decision=(u-b[:,:,token].float()).abs()<(u-a[:,:,token].float()).abs()
        carry=u-torch.where(decision,b[:,:,token],a[:,:,token]).float();oracle.append(decision)
    require(torch.equal(f1.v_code,ops._pack(torch.stack(oracle,2).transpose(2,3).contiguous())), "FB1 recurrence oracle")
    require(float(carry.abs().max())==s1["carry_terminal_maxabs"], "carry stats")
    tests.append("FB1_FP32_recurrence_and_stats")
    bad=source.clone();bad[0,0,0,0]=float("inf")
    try: ops.feedback_assign(state,bad,feedback=1,backend="torch")
    except ValueError: tests.append("nonfinite_source_rejected")
    else: raise ValueError("nonfinite source accepted")
    gradstate=copy(state);gradstate.v_scale=state.v_scale.clone().requires_grad_(True);gradstate.v_min=state.v_min.clone().requires_grad_(True)
    ops.decode_value_fields(gradstate).float().square().mean().backward()
    require(gradstate.v_scale.grad is not None and gradstate.v_min.grad is not None, "decode gradient path")
    tests.append("decode_preserves_metadata_gradient_path")
    # The exact frozen fold body can be exercised on CPU without importing its
    # unrelated CUDA bank dependency. Its source hash remains mandatory.
    fold_path=a2/"affine_rate_bank.py"
    require(sha(fold_path)==bridge._DEFAULT_SOURCES["a2/affine_rate_bank.py"], "frozen fold SHA")
    node=next(n for n in ast.parse(fold_path.read_text(encoding="utf-8")).body if isinstance(n,ast.FunctionDef) and n.name=="fold")
    env={"torch":torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(fold_path),"exec"),env)
    theta={"value_head_factors":torch.ones(28,4,requires_grad=True),"value_center_offsets":torch.zeros(28,4,requires_grad=True)}
    raw=(state.v_code,state.v_scale,state.v_min)
    out=ops.apply_theta_to_fields(raw,theta,fold_fn=env["fold"])
    expected=env["fold"](raw,1,theta["value_head_factors"][0][None,:,None,None],theta["value_center_offsets"][0][None,:,None,None])
    require(out[0] is state.v_code and all(torch.equal(x,y) for x,y in zip(out,expected)), "exact fold API")
    (out[1].float().sum()+out[2].float().sum()).backward()
    require(theta["value_head_factors"].grad is not None and theta["value_center_offsets"].grad is not None, "theta fold gradient")
    tests.append("frozen_fold_exact_and_theta_gradient")
    with bridge._legacy_paths(a2,a2.parent/"experiments",a2):
        new,stats=ops.reencode_value_residuals(state,source,[("k",0,0,0),("v",0,1,0),("v",0,1,1)],0)
    require(new is not state and new.v_code is state.v_code and new.v_scale is state.v_scale and new.v_min is state.v_min,"RE pure-return fields")
    require(torch.equal(new.depth_residual_code[0],state.depth_residual_code[0]) and torch.equal(new.depth_residual_alpha[0],state.depth_residual_alpha[0]),"RE K slot")
    require(stats["changed_code_elements"]+stats["changed_alpha_elements"]>0,"RE actually encoded")
    tests.append("hard_RE_changes_V_and_preserves_K_and_base")
    require(all(torch.equal(getattr(state,key),value) for key,value in snapshots.items()),"helper input mutated")
    tests.append("all_helpers_leave_original_state_unchanged")
    return {"status":"PASS_CPU_CONTROLS_ONLY", "tests":tests,"count":len(tests),"actual_GPU_calls":0,
            "GPU_equivalence":"PENDING_ROOT_EXECUTION", "frozen_fold_sha256":sha(fold_path),
            "CPU_RE_source_sha256":{name:sha(a2/name) for name in ("round6_depth_codec.py","quantizers.py")}}


def _same_state(left,right):
    return all(reference.tensor_identity(getattr(left,name))==reference.tensor_identity(getattr(right,name))
               for name in reference.STATE if hasattr(left,name) or hasattr(right,name))


def _k_fixed(ref, candidate, selected, layer):
    for name in ("direction_code","direction_gain","direction_mean","depth_residual_lookup","visual_pos"):
        require(torch.equal(getattr(ref,name),getattr(candidate,name)),"fixed K/lookup/position: "+name)
    for slot,(axis,index,head,stage) in enumerate(x for x in selected if x[1]==layer):
        if axis=="k":
            require(torch.equal(ref.depth_residual_code[slot],candidate.depth_residual_code[slot]) and
                    torch.equal(ref.depth_residual_alpha[slot],candidate.depth_residual_alpha[slot]),"fixed K residual")


@torch.no_grad()
def run_case(nv,config,variant,output):
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(0)
    native=DynamicCache();boundary=nv+4
    for layer in range(28):
        k=torch.randn(1,4,boundary,128,device="cuda",dtype=torch.float16)*.15
        v=torch.randn_like(k)*.1
        native.update(k,v,layer)
    k=v=None
    vp=torch.arange(4,boundary,device="cuda",dtype=torch.int64)
    initial=[(reference.tensor_identity(layer.keys),reference.tensor_identity(layer.values),id(layer)) for layer in native.layers]
    kwargs={name:config[name] for name in ("legacy_a2","legacy_experiments","legacy_scripts","kivi_source")}
    kwargs["expected_asset_sha256"]=reference.ASSETS
    original,metadata=bridge.build_r7_owners(native,vp,variant,**kwargs)
    expected_bytes=ops._unique(original)
    theta_json={"schema":ops.THETA_SCHEMA,"base_profile":"r7_joint_k1_gt_cond",
        "base_metadata_factor_sha256":ops.BASE_FACTOR_SHA,"parameter_count":224,
        "value_head_factors":(original[0].shared["encoder_factor.value_head_factors"]*1.25).cpu().tolist(),
        "value_center_offsets":(original[0].shared["encoder_factor.value_center_offsets"]+.2).cpu().tolist(),
        "fit_provenance":{"kind":"synthetic_nonidentity_control_fixture","GT_read":False,"fit":False}}
    theta_path=output/f"synthetic_theta_Nv{nv}.json";write(theta_path,theta_json)
    comparisons=[]
    for mode,theta_kind in (("original","original"),("V-FZ","original"),("V-RE","original"),
                            ("M-V","original"),("V-FZ","fixture"),("V-RE","fixture"),("FB0","fixture"),("FB1","fixture")):
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        candidates,info=ops.build_round9_value_owners(native,vp,variant,mode=mode,
            theta_path=theta_path if theta_kind=="fixture" else None,
            theta_sha256=sha(theta_path) if theta_kind=="fixture" else None,**kwargs)
        torch.cuda.synchronize()
        row={"mode":mode,"theta":theta_kind,"status":"RUNNING","selected_stage_count":info["selected_stage_count"],
             "complete_owner_core_bytes":ops._unique(candidates),"reference_owner_core_bytes":expected_bytes,
             "elapsed_seconds":time.perf_counter()-started,"GPU_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
             "candidate_metadata":info["round9_value_candidate"]}
        require(info["variant_sha256"]==metadata["variant_sha256"] and info["selected_stages"]==metadata["selected_stages"],"scope/ranking")
        require(ops._unique(candidates)==expected_bytes,"complete capacity identity")
        identical=True
        for layer,(ref,cand) in enumerate(zip(original,candidates)):
            _k_fixed(ref.state,cand.state,metadata["selected_stages"],layer)
            identical &= _same_state(ref.state,cand.state)
            require(torch.equal(ref.shared["book"],cand.shared["book"]) and torch.equal(ref.shared["encoder_factor.key_head_factors"],cand.shared["encoder_factor.key_head_factors"]),"fixed book/K factors")
            dense=ops.decode_value_fields(cand.state,nv)
            require(bool(torch.isfinite(dense).all()),"finite deployed V reconstruction")
            if mode=="V-FZ":
                require(torch.equal(ref.state.v_code,cand.state.v_code),"FZ cannot reassign base")
            if mode in ("M-V","V-FZ","FB0","FB1") and hasattr(ref.state,"depth_residual_code"):
                require(torch.equal(ref.state.depth_residual_code,cand.state.depth_residual_code) and torch.equal(ref.state.depth_residual_alpha,cand.state.depth_residual_alpha),"old residual must stay fixed")
        if theta_kind=="original" and mode in ("original","V-FZ","V-RE"):
            require(identical,"original theta full packed state identity")
            # Same frozen reader, exact text and query. No injected first token.
            reader_equal=[]
            qp=torch.arange(boundary,boundary+4,device="cuda",dtype=torch.long)
            tp=torch.cat((torch.arange(4,device="cuda"),qp))
            for layer,(ref,cand) in enumerate(zip(original,candidates)):
                tk=torch.cat((native.layers[layer].keys[:,:,:4],torch.randn(1,4,4,128,device="cuda",dtype=torch.float16)*.15),2)
                tv=torch.cat((native.layers[layer].values[:,:,:4],torch.randn(1,4,4,128,device="cuda",dtype=torch.float16)*.1),2)
                q=torch.randn(1,28,4,128,device="cuda",dtype=torch.float16)*.2
                args=dict(exact_keys=tk,exact_values=tv,exact_positions=tp,query_positions=qp,scale=128**-.5)
                left,right=ref.attend(q,**args),cand.attend(q,**args)
                require(torch.equal(left,right),"identity original reader bitwise")
                reader_equal.append(True)
            row["all28_reader_outputs_bitwise_equal"]=all(reader_equal)
        if mode=="V-RE" and theta_kind=="fixture":
            stats=info["round9_value_candidate"]["layer_stats"]
            changed=sum(s.get("changed_code_elements",0)+s.get("changed_alpha_elements",0) for s in stats)
            require(changed>0,"RE nonidentity fixture must actually change V code/alpha")
            row["actual_V_residual_changed_elements"]=changed
        if mode in ("FB0","FB1"):
            # Torch CPU recurrence is an execution-independent oracle for the
            # Triton serial scan. Check one non-residual and all V-residual heads
            # via two representative whole layers, not a fitted quality subset.
            probes=sorted({0,next(i for i,s in enumerate(candidates) if bool((s.state.depth_residual_lookup[1]>=0).any()))})
            oracle_results=[]
            for layer in probes:
                base=copy(candidates[layer].state)
                for key,value in vars(base).items():
                    if isinstance(value,torch.Tensor):setattr(base,key,value.detach().cpu())
                src=native.layers[layer].values.index_select(2,vp).cpu()
                oracle,_=ops.feedback_assign(base,src,feedback=int(mode[-1]),backend="torch")
                equal=torch.equal(oracle.v_code,candidates[layer].state.v_code.cpu())
                require(equal,"FB actual deployed-level CPU recurrence bitwise")
                oracle_results.append({"layer":layer,"base_codes_bitwise_equal":equal})
            row["execution_independent_FB_oracle"]=oracle_results
        require(initial==[(reference.tensor_identity(layer.keys),reference.tensor_identity(layer.values),id(layer)) for layer in native.layers],"native source pollution")
        row.update(status="PASS_SYNTHETIC_CONTROLS",all_K_and_K_residual_fixed=True,full_state_identity=identical,
                   native_source_unchanged=True,quality_claimed=False,numerical_tolerance_used=False)
        comparisons.append(row)
        candidates=None;gc.collect()
    return {"schema":SCHEMA+".case","status":"PASS_SYNTHETIC_CONTROLS","Nv":nv,
            "layers":28,"KV_heads":4,"query_heads":28,"head_dim":128,"seed":0,
            "reference_metadata":metadata,"modes":comparisons,"quality_QA_added":0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True,help="Frozen low gt_cond43 smoke4 config; supplies assets only")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpu-only",action="store_true")
    parser.add_argument("--legacy-a2-local",type=Path,default=Path("${FROZEN_TRAJKV_ROOT}/code/a2"))
    args=parser.parse_args()
    require(not args.output.exists(),"new unique output required")
    config=load(args.config);reference.validate_metadata(config)
    require(sha(ROOT/"code/r7_legacy_bridge_v1.py")==BRIDGE_SHA,"frozen bridge source")
    require(sha(ROOT/"code/r7_bridge_gpu_check_v1.py")==REFERENCE_SHA,"frozen reference helper")
    args.output.mkdir(parents=True)
    pins={str(Path(__file__).resolve()):sha(__file__),str(Path(ops.__file__).resolve()):sha(ops.__file__),
          str(args.config.resolve()):sha(args.config)}
    manifest={"schema":SCHEMA+".manifest","sources":pins,"config_sha256":sha(args.config),
              "synthetic_Nv":list(NV),"model_images_GT_loaded":False,"quality_QA_added":0,
              "scope":"B1 synthetic identity/fixed-format/FB recurrence, not model quality or gain evidence"}
    write(args.output/"manifest.json",manifest)
    rows=[]
    try:
        if args.cpu_only:
            receipt=cpu_controls(args.legacy_a2_local.resolve());write(args.output/"CPU_RECEIPT.json",receipt)
            result={"schema":SCHEMA+".complete",**receipt,"expected_GPU_cases":2,"recorded_GPU_cases":0}
        else:
            require(torch.cuda.is_available(),"actual CUDA required")
            require(torch.cuda.get_device_capability()==(8,6),"SM86 control required")
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            variant=reference.profile(config)
            with (args.output/"records.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
                for nv in NV:
                    row=run_case(nv,config,variant,args.output);rows.append(row)
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
            require(all(sha(path)==digest for path,digest in pins.items()),"source changed during controls")
            result={"schema":SCHEMA+".complete","status":"PASS_ALL_2_SYNTHETIC_B1_CONTROLS",
                    "expected_GPU_cases":2,"recorded_GPU_cases":len(rows),"quality_QA_added":0,
                    "records_sha256":sha(args.output/"records.jsonl"),"source_SHA_guard":True}
        write(args.output/"complete.json",result)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except Exception as exc:
        write(args.output/"FAILED.json",{"status":"FAILED","error":type(exc).__name__+": "+str(exc),"traceback":traceback.format_exc(),"actual_GPU_cases_written":len(rows)})
        raise


if __name__=="__main__":raise SystemExit(main())
