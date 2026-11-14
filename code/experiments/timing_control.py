"""Quiet single-GPU timing with Full-generated, identical forced token strings.

All generation/cache work goes through the frozen run_cycle.run_session path.
The Full transcript is an output-length control, not a correct-answer file.
Warmups are separate from randomized measurement rounds. No task quality is
scored, and forced text must never be reported as application accuracy.

Allocator constructor samples are collected in a separate unmeasured warmup
or dry repeat. Measured repeats have no constructor wrapper. Packed owner
bytes, post-constructor allocator increment, and whole-session peak are distinct
quantities; owned() excludes later repair snapshots and temporary workspace.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import io
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from bank import install_legacy
import run_cycle as runner_module
from run_cycle import digest, run_session


DEFAULT_POLICIES=['uniform','periodic','table_current','table_future','probe_l7_t0.02','margin_write']


def configurations(methods, policies):
    result=[]
    for supplied in methods:
        if ':' in supplied:
            method,policy=supplied.split(':',1)
            candidates=[policy]
        else:
            method=supplied
            candidates=policies if method.startswith('asym') else ['base']
        if not method or any(not policy for policy in candidates):
            raise ValueError('Method and policy names must be nonempty')
        for policy in candidates:
            entry=dict(method=method,policy=policy,configuration=f'{method}:{policy}')
            if entry not in result:
                result.append(entry)
    if not result:
        raise ValueError('At least one configuration is required')
    return result


def make_schedule(rows, configs, warmups, repeats, seed):
    schedule=[]
    for phase,rounds in (('warmup',warmups),('measurement',repeats)):
        for repeat in range(rounds):
            block=[dict(phase=phase,repeat=repeat,session_id=row['id'],**config)
                   for row in rows for config in configs]
            offset=0 if phase=='warmup' else 100000
            random.Random(seed+offset+repeat).shuffle(block)
            for trial in block:
                trial['trial_id']=f'trial_{len(schedule):05d}'
                schedule.append(trial)
    return schedule


def transcript_signature(turns):
    value=[dict(token_ids=turn['token_ids'],forced_chat_closure=turn['forced_chat_closure'],
                logical_end=turn['logical_end']) for turn in turns]
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_tokens(actual, reference):
    if len(actual)!=len(reference):
        raise ValueError('Forced run returned a different number of turns')
    for number,(current,target) in enumerate(zip(actual,reference),1):
        for field in ('token_ids','forced_chat_closure','logical_end'):
            if current[field]!=target[field]:
                raise ValueError(f'Forced run changed {field} at turn {number}')
    return dict(output_token_ids_identical=True,chat_closures_identical=True,
                logical_positions_identical=True,sha256=transcript_signature(actual))


def logical_reads(events):
    fields=('base_visual_bytes','residual_bytes','text_bytes','reference_bytes')
    totals={field:sum(int(row.get(field,0)) for row in events) for field in fields}
    by_pass={}
    for row in events:
        target=by_pass.setdefault(row['pass_kind'],dict(calls=0,query_tokens=0,**{key:0 for key in fields}))
        for key in ('calls','query_tokens',*fields):
            target[key]+=int(row.get(key,0))
    for target in by_pass.values():
        target['total_bytes']=sum(target[field] for field in fields)
    probe=sum(row['total_bytes'] for key,row in by_pass.items() if key=='probe')
    replay=sum(row['total_bytes'] for key,row in by_pass.items() if key in ('replay','repair'))
    extra=sum(int(row.get('residual_bytes',0))+
              (int(row.get('base_visual_bytes',0))+int(row.get('text_bytes',0))
               if row['pass_kind'] in ('probe','replay','repair') else 0) for row in events)
    return dict(**totals,total_logical_read_bytes=sum(totals.values()),
                probe_total_bytes=probe,replay_or_repair_total_bytes=replay,
                extra_logical_read_bytes=extra,by_pass_kind=by_pass,
                scope='actual fused-call logical tensor payloads, not measured HBM transactions',
                overlap='probe/replay totals overlap base/residual/text totals; do not sum these categories together')


def timing_metrics(result):
    intervals=0
    weighted=0.
    per_turn=[]
    for turn in result['turns']:
        count=max(len(turn['token_ids'])-1,0)
        value=turn.get('mean_inter_token_ms')
        if count and value is not None:
            weighted+=count*value
            intervals+=count
        per_turn.append(dict(turn=turn['turn'],output_tokens=len(turn['token_ids']),
                             ttft_ms=turn.get('ttft_ms'),session_turn_ms=turn['engine_ms'],
                             mean_inter_token_ms=value if count else None,inter_token_intervals=count))
    return dict(session_ms=result['session_ms'],first_turn_ttft_ms=per_turn[0]['ttft_ms'] if per_turn else None,
                per_turn=per_turn,mean_inter_token_ms=weighted/intervals if intervals else None,
                inter_token_intervals=intervals,output_tokens=sum(len(t['token_ids']) for t in result['turns']),
                inter_token_scope='weighted mean of runner per-turn interval means; individual interval distribution is not exposed',
                ttft_scope='first turn includes input preparation, native visual prefill and cache encoding; later turns include text suffix processing',
                timing_scope='unchanged run_session timers; identical forced token schedule and same cache/consume path')


def quiet_snapshot():
    """Observe all NVIDIA compute processes, including other physical GPUs."""
    try:
        completed=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=False)
    except (OSError,subprocess.SubprocessError) as exc:
        return dict(verified=False,quiet=False,error=type(exc).__name__+': '+str(exc),processes=[])
    if completed.returncode:
        return dict(verified=False,quiet=False,error=completed.stderr.strip(),processes=[])
    processes=[]
    for line in completed.stdout.splitlines():
        parts=[part.strip() for part in line.split(',',2)]
        if len(parts)!=3:
            continue
        try:
            pid=int(parts[0])
        except ValueError:
            continue
        processes.append(dict(pid=pid,gpu_uuid=parts[1],process_name=parts[2],self=pid==os.getpid()))
    foreign=[process for process in processes if not process['self']]
    return dict(verified=True,quiet=not foreign,processes=processes,foreign_processes=foreign,
                scope='NVIDIA compute-process inventory at boundary; not continuous OS/thermal isolation')


def require_quiet(args):
    snapshot=quiet_snapshot()
    if not snapshot['quiet'] and not args.allow_busy:
        raise RuntimeError('Quiet timing requires no other NVIDIA compute process on any GPU. '+json.dumps(snapshot))
    return snapshot


def _stats(values):
    values=[float(value) for value in values if value is not None]
    return dict(n=len(values),median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None)


def allocator_point():
    """Synchronized allocator sample; used only outside measured repeats."""
    torch.cuda.synchronize()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved())


def native_cache_owner_bytes(cache):
    # Return scalar ownership metadata only. Retaining a pre-constructor tensor
    # here would keep FP16 KV alive and invalidate the post-encoding sample.
    owners={}
    for layer in cache.layers:
        for name in ('keys','values'):
            tensor=getattr(layer,name,None)
            if tensor is not None:
                storage=tensor.untyped_storage()
                owners[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return dict(unique_bytes=sum(owners.values()),unique_storages=len(owners),
                scope='combined native cache tensor storages immediately before bank conversion')


@contextlib.contextmanager
def observe_bank_constructor(observations):
    """Temporarily observe metadata around the unchanged bank constructor.

    Use only in unmeasured warmup/dry repeats. Sampling synchronizes CUDA and
    calls owned(), so it must not be installed during reported timing repeats.
    The constructor is called exactly once, with identical arguments, and the
    observer retains no cache tensors or bank references.
    """
    original=runner_module.ProgressiveBank
    def construct(*arguments,**keywords):
        cache=keywords.get('cache',arguments[1] if len(arguments)>1 else None)
        if cache is None:
            raise ValueError('Cannot locate native cache in bank constructor arguments')
        before=allocator_point()
        native_owner=native_cache_owner_bytes(cache)
        bank=original(*arguments,**keywords)
        after=allocator_point()
        owned=bank.owned()
        observations.append(dict(before_constructor=before,after_constructor=after,
                                 native_cache_before=native_owner,bank_owned_after=owned,
                                 owned_unique_bytes=owned['total_unique'],
                                 constructor_allocated_change_bytes=after['allocated_bytes']-before['allocated_bytes'],
                                 constructor_reserved_change_bytes=after['reserved_bytes']-before['reserved_bytes']))
        return bank
    runner_module.ProgressiveBank=construct
    try:
        yield
    finally:
        runner_module.ProgressiveBank=original


def allocator_audit(audit_id, trial, result, observed):
    if observed is None or len(observed['constructors'])!=1:
        raise ValueError('Allocator audit expected exactly one observed bank constructor')
    constructor=observed['constructors'][0]
    baseline=observed['model_only_before_runner']
    after=constructor['after_constructor']
    return dict(audit_id=audit_id,session_id=trial['session_id'],configuration=trial['configuration'],
                model_only_before_runner=baseline,**constructor,
                steady_allocated_increment_from_model_only_bytes=after['allocated_bytes']-baseline['allocated_bytes'],
                steady_reserved_increment_from_model_only_bytes=after['reserved_bytes']-baseline['reserved_bytes'],
                session_peak_allocated_bytes=result['peak_allocated'],
                session_peak_reserved_bytes=observed.get('session_peak_reserved_bytes'),
                session_peak_minus_model_only_bytes=result['peak_allocated']-baseline['allocated_bytes'],
                collection_phase='separate unmeasured warmup/dry repeat',
                steady_scope='synchronized post-constructor allocator delta; includes other live setup tensors, not an assertion of bank-only ownership',
                owned_scope='bank.owned unique storages only; excludes repair_start snapshots and other later temporary workspaces',
                peak_scope='whole dry-repeat allocator peak includes prefill, encoding, repair snapshots, probe/replay and temporary workspaces',
                timing_relation='allocator sampling and owned() are disabled in measured timing repeats')


def build_summary(records, configs, selected_ids, repeats):
    values=[]
    for config in configs:
        chosen=[record for record in records if record['configuration']==config['configuration']]
        per_session=[]
        for session in selected_ids:
            current=[record for record in chosen if record['session_id']==session]
            per_session.append(dict(session_id=session,measured_repeats=len(current),
                                    session_ms=_stats([r['timing']['session_ms'] for r in current]),
                                    first_turn_ttft_ms=_stats([r['timing']['first_turn_ttft_ms'] for r in current]),
                                    mean_inter_token_ms=_stats([r['timing']['mean_inter_token_ms'] for r in current]),
                                    peak_allocated_bytes=_stats([r['memory']['peak_allocated_bytes'] for r in current]),
                                    allocator_audit_ids=sorted({r['memory_allocator_audit']['audit_id'] for r in current}),
                                    owned_final=[r['owned_final'] for r in current]))
        repeat_totals=[]
        for repeat in range(repeats):
            current=[record for record in chosen if record['repeat']==repeat]
            repeat_totals.append(dict(repeat=repeat,sessions=len(current),
                                     session_ms_sum=sum(r['timing']['session_ms'] for r in current),
                                     output_tokens=sum(r['timing']['output_tokens'] for r in current),
                                     all_quiet=bool(current) and all(r['quiet_verified'] for r in current)))
        values.append(dict(**config,measured_trials=len(chosen),expected_trials=len(selected_ids)*repeats,
                           complete=len(chosen)==len(selected_ids)*repeats,
                           all_tokens_identical=bool(chosen) and all(r['token_identity']['output_token_ids_identical'] for r in chosen),
                           all_quiet=bool(chosen) and all(r['quiet_verified'] for r in chosen),
                           read_budget_ok=bool(chosen) and all(r['read_budget_ok'] for r in chosen),
                           repeat_totals=repeat_totals,per_session=per_session))
    return dict(schema='trajkv.timing_control_summary.v1',quality='not_scored',
                application_accuracy=None,teacher_forced_tokens_are_not_quality_results=True,
                warmups_included=False,configurations=values,
                interpretation='instrumented same-output timing and memory; no free-running quality or speedup claim is inferred')


def _call(model,processor,row,runner_args,method,policy,forced,observe_allocator=False):
    captured=io.StringIO()
    observed=None
    if observe_allocator:
        gc.collect();torch.cuda.empty_cache()
        observed=dict(model_only_before_runner=allocator_point(),constructors=[])
    observer=observe_bank_constructor(observed['constructors']) if observed is not None else contextlib.nullcontext()
    started=time.perf_counter()
    with contextlib.redirect_stdout(captured),observer:
        result,states=run_session(model,processor,row,runner_args,method,policy,forced=forced,keep_reference=False)
    elapsed=1000*(time.perf_counter()-started)
    if states:
        raise ValueError('Timing control must not capture diagnostic state snapshots')
    if observed is not None:
        observed['session_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    return result,captured.getvalue(),elapsed,observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','samples','legacy-scripts','kivi-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--methods',nargs='+',default=['full','base2','native4','asym'])
    parser.add_argument('--policies',nargs='+',default=DEFAULT_POLICIES)
    parser.add_argument('--gain-table',type=Path)
    parser.add_argument('--read-budget-bytes',type=int)
    parser.add_argument('--count',type=int,default=4);parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--turns',type=int,default=10);parser.add_argument('--max-tokens',type=int,default=24)
    parser.add_argument('--max-patches',type=int,default=4)
    parser.add_argument('--description-tokens',type=int,default=96)
    parser.add_argument('--hide-visual-after-description',action='store_true')
    parser.add_argument('--margin',type=float,default=1.);parser.add_argument('--max-replays',type=int,default=2)
    parser.add_argument('--probe-layer',type=int,default=7);parser.add_argument('--probe-threshold',type=float,default=.02)
    parser.add_argument('--warmups',type=int,default=1);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--allow-busy',action='store_true',help='Diagnostic override; such runs are never marked quiet')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.count,args.turns,args.max_tokens,args.max_patches,args.description_tokens,args.repeats)<1 or args.warmups<0:
        parser.error('Counts and repeats must be positive; warmups must be nonnegative')
    if args.read_budget_bytes is not None and args.read_budget_bytes<0:
        parser.error('Read budget must be nonnegative')
    configs=configurations(args.methods,args.policies)
    gain_data=json.loads(args.gain_table.read_text(encoding='utf-8')) if args.gain_table else None
    if any(config['policy'].startswith('table_') for config in configs) and gain_data is None:
        parser.error('Table policies require --gain-table pointing to frozen calibration data')
    rows=json.loads(args.samples.read_text(encoding='utf-8'))[args.start:args.start+args.count]
    if len(rows)!=args.count or len({row['id'] for row in rows})!=len(rows):
        parser.error('Selected development sessions must be present and have unique IDs')
    for row in rows:
        if row.get('split') not in ('development','dev','calibration'):
            parser.error('Quiet timing is restricted to development/calibration sessions')
        if any(key in row for key in ('answers','reference_answers','gold_answers','ground_truth')):
            parser.error('Runner samples must not contain reference answers')
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES for quiet timing')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    initial_quiet=require_quiet(args)
    install_legacy(args.legacy_scripts)
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    # Populate every current runner field, retaining getattr-compatible defaults
    # for policies whose additional options may not be exercised by this run.
    runner_args=SimpleNamespace(**vars(args))
    runner_args.gain_table_data=gain_data
    runner_args.save_states=False
    runner_args.reference=None
    schedule=make_schedule(rows,configs,args.warmups,args.repeats,args.seed)
    local=Path(__file__).parent
    sources=[Path(__file__),local/'run_cycle.py',local/'bank.py',args.samples,
             args.legacy_scripts/'p202_fused.py',args.kivi_source]
    for optional in ('self_probe.py','fused_progressive.py'):
        if (local/optional).exists():sources.append(local/optional)
    if args.gain_table:sources.append(args.gain_table)
    source_hashes={str(path):digest(path) for path in sources}
    contract=dict(configurations=configs,selected_ids=[row['id'] for row in rows],turns=args.turns,
                  max_tokens=args.max_tokens,max_patches=args.max_patches,description_tokens=args.description_tokens,
                  read_budget_bytes=args.read_budget_bytes,gain_table_sha256=digest(args.gain_table) if args.gain_table else None,
                  seed=args.seed,warmups=args.warmups,repeats=args.repeats,sources=source_hashes)
    contract['runner_options']={key:str(value) if isinstance(value,Path) else value
                                for key,value in vars(args).items()
                                if key not in ('output','resume','allow_busy')}
    manifest_path=args.output/'manifest.json'
    if args.resume and manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['contract']!=contract:
            raise ValueError('Resume contract/source hashes differ; use a new timing output directory')
    else:
        manifest=dict(contract=contract,arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      torch=torch.__version__,gpu=torch.cuda.get_device_name(),initial_quiet=initial_quiet,
                      quality='not_scored',reference_answers_visible_to_runner=False,paid_calls=0,
                      canonical_source='Full free generation solely to freeze output tokens',
                      full_baseline_scope='full FP16 P202 fused bank after native visual prefill; native SDPA parity is a separate audit')
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (args.output/'schedule.json').write_text(json.dumps(schedule,indent=2),encoding='utf-8')
        snapshot=args.output/'source_snapshot';snapshot.mkdir()
        for number,path in enumerate(sources):shutil.copy2(path,snapshot/f'{number:02d}_{path.name}')
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,
                device_map='cuda:0',attn_implementation='sdpa').eval()
    canonical_path=args.output/'canonical_full.jsonl'
    canonical={}
    if args.resume and canonical_path.exists():
        canonical={record['session_id']:record for record in map(json.loads,canonical_path.read_text(encoding='utf-8').splitlines()) if record}
    with torch.inference_mode(),canonical_path.open('a',encoding='utf-8') as log:
        for row in rows:
            if row['id'] in canonical:continue
            require_quiet(args)
            result,stdout,elapsed,_=_call(model,processor,row,runner_args,'full','base',None)
            record=dict(session_id=row['id'],quality='not_scored',canonical_full=result,
                        token_signature=transcript_signature(result['turns']),outer_wall_ms=elapsed)
            log.write(json.dumps(record)+'\n');log.flush();canonical[row['id']]=record
            (args.output/f'canonical_{rows.index(row):03d}_stdout.txt').write_text(stdout,encoding='utf-8')
            print(json.dumps(dict(stage='freeze_full_trajectory',session=row['id'],
                                  tokens=sum(len(turn['token_ids']) for turn in result['turns']))),flush=True)
    seen=set();measurements=[]
    allocator_path=args.output/'allocator_audits.jsonl'
    allocator_records={}
    if args.resume and allocator_path.exists():
        for line in allocator_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                audit=json.loads(line)
                allocator_records[(audit['session_id'],audit['configuration'])]=audit
    for filename,measured in (('warmup_records.jsonl',False),('records.jsonl',True)):
        path=args.output/filename
        if args.resume and path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                record=json.loads(line);seen.add(record['trial_id'])
                if measured:measurements.append(record)
    by_id={row['id']:row for row in rows}
    trace_dir=args.output/'trial_stdout';trace_dir.mkdir(exist_ok=True)
    invocation=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with torch.inference_mode(),(args.output/'warmup_records.jsonl').open('a',encoding='utf-8') as warm_log, \
         (args.output/'records.jsonl').open('a',encoding='utf-8') as measured_log, \
         allocator_path.open('a',encoding='utf-8') as allocator_log:
        for trial in schedule:
            if trial['trial_id'] in seen:continue
            before=require_quiet(args)
            # Prevent late source edits from silently mixing implementations in
            # an otherwise identically named timing table.
            for path,expected in source_hashes.items():
                if digest(path)!=expected:raise RuntimeError('Timing source changed during run: '+path)
            row=by_id[trial['session_id']]
            reference=canonical[row['id']]['canonical_full']['turns']
            audit_key=(row['id'],trial['configuration'])
            if trial['phase']=='measurement' and audit_key not in allocator_records:
                # With zero warmups (or an interrupted audit), collect a separate
                # unmeasured repeat instead of instrumenting a measured trial.
                dry,trace,_,observed=_call(model,processor,row,runner_args,trial['method'],trial['policy'],reference,True)
                verify_tokens(dry['turns'],reference)
                audit=allocator_audit('allocator_'+trial['trial_id'],trial,dry,observed)
                allocator_log.write(json.dumps(audit)+'\n');allocator_log.flush()
                allocator_records[audit_key]=audit
                (trace_dir/f'allocator_{trial["trial_id"]}.txt').write_text(trace,encoding='utf-8')
                before=require_quiet(args)
            baseline_allocated=torch.cuda.memory_allocated()
            collect_allocator=trial['phase']=='warmup' and audit_key not in allocator_records
            result,stdout,outer_ms,observed=_call(model,processor,row,runner_args,trial['method'],trial['policy'],reference,collect_allocator)
            token_identity=verify_tokens(result['turns'],reference)
            if observed is not None:
                audit=allocator_audit('allocator_'+trial['trial_id'],trial,result,observed)
                allocator_log.write(json.dumps(audit)+'\n');allocator_log.flush()
                allocator_records[audit_key]=audit
            reads=logical_reads(result['read_events'])
            if reads['extra_logical_read_bytes']!=result.get('extra_logical_read_bytes',reads['extra_logical_read_bytes']):
                raise ValueError('Timing read ledger does not reconcile with runner')
            after=quiet_snapshot()
            quiet=bool(before['quiet'] and after['quiet'])
            record=dict(schema='trajkv.timing_control.v1',**trial,invocation_utc=invocation,
                        quality='not_scored',application_accuracy=None,
                        token_identity=token_identity,timing=timing_metrics(result),
                        outer_wall_ms_including_runner_cleanup=outer_ms,
                        memory=dict(peak_allocated_bytes=result['peak_allocated'],
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    allocated_before_runner=baseline_allocated,
                                    peak_allocated_minus_before=result['peak_allocated']-baseline_allocated),
                        memory_allocator_audit=allocator_records[audit_key],
                        owned_initial=result['owned_initial'],owned_final=result['owned_final'],
                        logical_reads=reads,read_budget_ok=result.get('read_budget_ok',True),
                        read_ledger_exact=result.get('read_ledger_exact'),quiet_verified=quiet,
                        quiet_before=before,quiet_after=after,runner_result=result,
                        teacher_forced_output_is_not_application_quality=True,paid_calls=0)
            target=warm_log if trial['phase']=='warmup' else measured_log
            target.write(json.dumps(record)+'\n');target.flush()
            (trace_dir/f'{trial["trial_id"]}.txt').write_text(stdout,encoding='utf-8')
            if trial['phase']=='measurement':measurements.append(record)
            print(json.dumps(dict(stage=trial['phase'],trial=trial['trial_id'],repeat=trial['repeat'],
                                  session=row['id'],configuration=trial['configuration'],
                                  session_ms=result['session_ms'],ttft_ms=record['timing']['first_turn_ttft_ms'],
                                  quiet_verified=quiet,tokens_identical=True)),flush=True)
            if not quiet and not args.allow_busy:
                raise RuntimeError('Another compute process appeared during timing; trial preserved and marked nonquiet')
    summary=build_summary(measurements,configs,[row['id'] for row in rows],args.repeats)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(measured_trials=len(measurements),
                    expected_measurements=len(rows)*len(configs)*args.repeats,quality='not_scored',paid_calls=0)),encoding='utf-8')


if __name__=='__main__':
    main()
