"""Root-run controls for new Round9 K-book and read/storage prefix views.

Only synthetic K/V, no model/images/GT/fitting. --cpu-only exercises the actual
TIER transform with explicitly fake packed owners; CUDA reader evidence remains
pending until Root executes the default two-Nv SM86 control.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
import torch
import r7_legacy_bridge_v1 as bridge
import r7_bridge_gpu_check_v1 as ref
import round9_extended_bank_v1 as extended
from ownership import storage_report

SCHEMA="trajkv.round9.extended_synthetic_control.v1"
NV=(256,1792)
BRIDGE_SHA="b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b"
BRIDGE2_SHA="608b1660896c8e8fd05484be47acf73e99f260db71731239937d0279ed842dad"
REFERENCE_SHA="e37810a2568ebb21f10ea10488829be0758311985c5dbdeebdd20fa450c5631a"


def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
def physical(owners):return storage_report(ref.candidate_named(owners))


def _fake_original(selected,nv=32):
    """Only a CPU structural fixture; never passed as an actual R7 observation."""
    lookup=torch.full((28,2,4,2),-1,dtype=torch.int32)
    vp=torch.arange(4,nv+4)
    shared={"book":torch.zeros(16,4,dtype=torch.float16),"global_residual_lookup":lookup,
            "encoder_ranking_bytes":torch.zeros(448,dtype=torch.uint8),
            "encoder_policy_bytes":torch.zeros(5,dtype=torch.uint8)}
    for name in ("key_head_factors","value_head_factors","value_center_offsets"):
        shared["encoder_factor."+name]=torch.ones(28,4)
    owners=[]
    for layer in range(28):
        rows=[item for item in selected if item[1]==layer]
        state=SimpleNamespace(direction_code=torch.zeros(1,4,nv,16,dtype=torch.uint8),
            direction_gain=torch.zeros(1,4,nv,dtype=torch.float16),direction_mean=torch.zeros(1,4,1,128,dtype=torch.float16),
            v_code=torch.zeros(1,4,128,nv//32,dtype=torch.int32),
            v_scale=torch.zeros(1,4,128,1,dtype=torch.float16),v_min=torch.zeros(1,4,128,1,dtype=torch.float16),
            visual_pos=vp,depth_residual_lookup=lookup[layer])
        if rows:
            state.depth_residual_code=torch.arange(len(rows)*nv*4,dtype=torch.int32).reshape(len(rows),nv,4)
            state.depth_residual_alpha=torch.arange(len(rows)*nv,dtype=torch.float16).reshape(len(rows),nv)
            for slot,(axis,_,head,stage) in enumerate(rows):lookup[layer,0 if axis=="k" else 1,head,stage]=slot
        owners.append(bridge.R7LegacyOwner(layer,state,shared,None,nv+4))
    return owners,{"selected_stages":deepcopy(selected),"selected_stage_count":43,
                   "returned_owners_unique_tensor_capacity_bytes":extended._capacity(owners)}


def _active_values(owners):
    result={}
    for layer,owner in enumerate(owners):
        state=owner.state
        for axis in range(2):
            for head in range(4):
                for stage in range(2):
                    slot=int(state.depth_residual_lookup[axis,head,stage])
                    if slot>=0:
                        result[(layer,axis,head,stage)]=(ref.tensor_identity(state.depth_residual_code[slot]),
                                                       ref.tensor_identity(state.depth_residual_alpha[slot]))
    return result


def cpu_controls(config):
    selected=config["variant"]["selected_stages"][:43]
    require(len(selected)==43,"original43 fixture")
    expected,_=_fake_original(selected)
    baseline=physical(expected)["total_bytes"]
    cases=[]
    for count in (0,21,43):
        snapshots={}
        for axis in ("read","storage"):
            def fresh(*args,**kwargs):return _fake_original(selected)
            with patch("r7_legacy_bridge_v2.build_r7_owners",side_effect=fresh):
                owners,meta=extended.build_round9_extended_owners(SimpleNamespace(),torch.arange(32),
                    {"name":"r7_joint_k1_gt_cond"},candidate={"mode":"TIER","tier_count":count,"tier_axis":axis})
            report=physical(owners)
            saved=(43-count)*18*32 if axis=="storage" else 0
            require(report["total_bytes"]==baseline-saved,"actual CPU storage capacity difference")
            require(meta["selected_stages"]==selected[:count] and meta["selected_stage_count"]==count,"prefix scope")
            snapshots[axis]=_active_values(owners)
            if count==0 and axis=="storage":
                require(all(o.state.depth_residual_code.numel()==o.state.depth_residual_alpha.numel()==0 for o in owners),"m0 empty pools")
            cases.append({"m":count,"axis":axis,"status":"PASS_FAKE_OWNER_STRUCTURE_ONLY",
                          "actual_unique_bytes":report["total_bytes"],"saved_bytes":baseline-report["total_bytes"],
                          "expected_saved_bytes":saved})
        require(snapshots["read"]==snapshots["storage"],"remapped pools preserve active packed values")
    return {"status":"PASS_CPU_FAKE_OWNER_TIER_CONTROLS_ONLY","cases":cases,
            "actual_GPU_calls":0,"actual_legacy_codec_runs":0,"GPU_reader":"PENDING_ROOT_EXECUTION",
            "fixture_warning":"No fake values are method results or evidence of CUDA equivalence"}


def _state_equal(left,right):
    return all(ref.compare_tensor(getattr(left,name,None),getattr(right,name,None))["bitwise_equal"] for name in ref.STATE)


def _read_same(owners,queries,text):
    values=[]
    for owner,q,args in zip(owners,queries,text):
        result=owner.attend(q,**args)
        require(bool(torch.isfinite(result).all()),"finite actual packed reader")
        require(owner.state.text_k.numel()==owner.state.text_v.numel()==owner.state.text_pos.numel()==0,"borrowed text released")
        values.append(result)
    return values


@torch.no_grad()
def run_case(nv,config,variant,output):
    from transformers.cache_utils import DynamicCache
    from r7_legacy_bridge_v2 import build_r7_owners
    torch.manual_seed(0)
    native=DynamicCache();boundary=nv+4
    for layer in range(28):
        key=torch.randn(1,4,boundary,128,device="cuda",dtype=torch.float16)*.15
        value=torch.randn_like(key)*.1;native.update(key,value,layer)
    key=value=None
    vp=torch.arange(4,boundary,device="cuda",dtype=torch.int64)
    initial=[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values),id(x)) for x in native.layers]
    kwargs={name:config[name] for name in ("legacy_a2","legacy_experiments","legacy_scripts","kivi_source")}
    kwargs["expected_asset_sha256"]=ref.ASSETS
    original,base_metadata=build_r7_owners(native,vp,variant,**kwargs)
    require(base_metadata["selected_stage_count"]==43,"actual43")
    before=physical(original)
    require(before["total_bytes"]==base_metadata["legacy_declared_complete_visual_bytes"],"frozen reference owner bytes")
    qp=torch.arange(boundary,boundary+4,device="cuda",dtype=torch.int64)
    tp=torch.cat((torch.arange(4,device="cuda",dtype=torch.int64),qp))
    queries=[];text=[]
    for layer in native.layers:
        queries.append(torch.randn(1,28,4,128,device="cuda",dtype=torch.float16)*.2)
        text.append(dict(exact_keys=torch.cat((layer.keys[:,:,:4],torch.randn(1,4,4,128,device="cuda",dtype=torch.float16)*.15),2),
                         exact_values=torch.cat((layer.values[:,:,:4],torch.randn(1,4,4,128,device="cuda",dtype=torch.float16)*.1),2),
                         exact_positions=tp,query_positions=qp,scale=128**-.5))
    baseline_output=_read_same(original,queries,text)
    book_path=output/f"original_book_identity_Nv{nv}.json"
    write(book_path,{"schema":"trajkv.round9.book_synthetic_identity.v1",
                    "original_book_sha256":ref.ASSETS["direction_book_path"],
                    "centers":original[0].shared["book"].cpu().tolist(),"fit":False})
    identity,identity_meta=extended.build_round9_extended_owners(native,vp,variant,
        candidate={"mode":"K-MSE","book_path":str(book_path),"book_sha256":sha(book_path)},**kwargs)
    require(all(_state_equal(x.state,y.state) for x,y in zip(original,identity)),"original K book reencode packed identity")
    require(torch.equal(original[0].shared["book"],identity[0].shared["book"]),"original book values")
    require(physical(identity)["total_bytes"]==before["total_bytes"],"K book capacity identity")
    actual=_read_same(identity,queries,text)
    require(all(torch.equal(a,b) for a,b in zip(baseline_output,actual)),"K identity reader bitwise")
    identity_record={"status":"PASS_ORIGINAL_BOOK_REENCODE_IDENTITY","all28_packed_state_bitwise":True,
                     "all28_reader_outputs_bitwise":True,"complete_bytes":physical(identity)["total_bytes"],
                     "candidate_metadata":identity_meta}
    identity=actual=None;gc.collect()
    tiers=[]
    for count in (0,21,43):
        stored=[]
        outputs={};active={}
        for axis in ("read","storage"):
            started=time.perf_counter()
            owners,meta=extended.build_round9_extended_owners(native,vp,variant,
                candidate={"mode":"TIER","tier_count":count,"tier_axis":axis},**kwargs)
            report=physical(owners)
            expected_saved=(43-count)*18*nv if axis=="storage" else 0
            require(before["total_bytes"]-report["total_bytes"]==expected_saved,"strict physical difference18Nv per unit")
            require(meta["selected_stages"]==base_metadata["selected_stages"][:count],"prefix fixed by original priority")
            active[axis]=_active_values(owners)
            outputs[axis]=_read_same(owners,queries,text)
            if count==43:require(all(torch.equal(a,b) for a,b in zip(outputs[axis],baseline_output)),"m43 full reader identity")
            storage_code_bytes=sum(o.state.depth_residual_code.untyped_storage().nbytes() for o in owners
                                   if hasattr(o.state,"depth_residual_code"))
            storage_alpha_bytes=sum(o.state.depth_residual_alpha.untyped_storage().nbytes() for o in owners
                                    if hasattr(o.state,"depth_residual_alpha"))
            require(storage_code_bytes+storage_alpha_bytes==(43 if axis=="read" else count)*18*nv,"complete residual pool capacities")
            stored.append({"axis":axis,"status":"PASS_ACTUAL_PACKED_TIER","complete_unique_bytes":report["total_bytes"],
                "actual_saved_bytes":before["total_bytes"]-report["total_bytes"],"expected_saved_bytes":expected_saved,
                "pool_code_bytes":storage_code_bytes,"pool_alpha_bytes":storage_alpha_bytes,
                "all28_outputs_finite":True,"physical_storage_report":report,"metadata":meta,
                "host_elapsed_seconds":time.perf_counter()-started})
            owners=None;gc.collect()
        require(active["read"]==active["storage"],"active old/resliced packed data identity")
        require(all(torch.equal(a,b) for a,b in zip(outputs["read"],outputs["storage"])),"read/storage sameq output bitwise")
        tiers.append({"m":count,"read_storage_all28_outputs_bitwise_equal":True,
                      "active_code_alpha_bitwise_equal":True,"cases":stored})
    require(initial==[(ref.tensor_identity(x.keys),ref.tensor_identity(x.values),id(x)) for x in native.layers],"original native source mutated")
    return {"schema":SCHEMA+".case","status":"PASS_ACTUAL_EXTENDED_CONTROLS","Nv":nv,
            "K_original_book_identity":identity_record,"tiers":tiers,"reference_storage":before,
            "all_native_sources_unchanged":True,"numerical_tolerance_used":False,"quality_QA_added":0,
            "diagnostic_storage_not_deployment_savings":True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--cpu-only",action="store_true")
    args=parser.parse_args();config=load(args.config);ref.validate_metadata(config)
    require(not args.output.exists(),"new unique output required")
    for path,digest in (("r7_legacy_bridge_v1.py",BRIDGE_SHA),("r7_legacy_bridge_v2.py",BRIDGE2_SHA),("r7_bridge_gpu_check_v1.py",REFERENCE_SHA)):
        require(sha(ROOT/"code"/path)==digest,"frozen source changed: "+path)
    sources={str(Path(path).resolve()):sha(path) for path in (__file__,extended.__file__,args.config)}
    args.output.mkdir(parents=True)
    write(args.output/"manifest.json",{"schema":SCHEMA+".manifest","sources":sources,"Nv":list(NV),
        "model_images_GT_loaded":False,"quality_QA_added":0,"read_storage_m":[0,21,43],
        "scope":"K original identity and real storage/read prefix control, not quality or speed evidence"})
    cases=[]
    try:
        if args.cpu_only:
            result={"schema":SCHEMA+".complete",**cpu_controls(config)}
            write(args.output/"CPU_RECEIPT.json",result)
        else:
            require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6),"actual SM86 CUDA required")
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            variant=ref.profile(config)
            with (args.output/"records.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
                for nv in NV:
                    item=run_case(nv,config,variant,args.output);cases.append(item)
                    stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
            result={"schema":SCHEMA+".complete","status":"PASS_ALL_2_ACTUAL_EXTENDED_SYNTHETIC_CONTROLS",
                    "expected_GPU_cases":2,"recorded_GPU_cases":len(cases),"records_sha256":sha(args.output/"records.jsonl"),
                    "quality_QA_added":0,"numerical_tolerance_used":False}
        require(all(sha(path)==digest for path,digest in sources.items()),"source changed during controls")
        write(args.output/"complete.json",result);print(json.dumps(result));return 0
    except Exception as exc:
        write(args.output/"FAILED.json",{"status":"FAILED","error":type(exc).__name__+": "+str(exc),"traceback":traceback.format_exc(),
                                        "GPU_cases_completed":len(cases)})
        raise


if __name__=="__main__":raise SystemExit(main())
