"""Single-resident-model, interleaved, streaming end-to-end speed benchmark.

CPU --plan imports no CUDA/model package. Actual timing preserves the native
image-prefix / full-question-suffix protocol and real packed method readers.
This driver does not edit evaluate.py, its allowlist, or any method source.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager,nullcontext
from copy import deepcopy
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=Path(os.environ.get('BASELINE_DATA_ROOT',str(ROOT)))
DEFAULT_SUFFIX='\nAnswer the question using a single word or phrase.'
JOB_FIELDS={'schema','eval_configs','inputs','ids','selection','repeat','warmup_per_qa','seed',
            'expected_predictions','allocator_policy','verify_gpu_exclusive','host_quiet_asserted',
            'adapter_extensions'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def resolve_path(value,relative_to):
    path=Path(value)
    return path if path.is_absolute() else Path(relative_to)/path


def extension_registry(job):
    """Driver-local exact-source registry, never a core allowlist mutation."""
    registry={}
    for item in job.get('adapter_extensions',[]):
        require(set(item)=={'module','class','source_files','support_receipt','support_receipt_sha256'},'Invalid explicit extension registry entry')
        pair=(item['module'],item['class'])
        require(re.fullmatch(r'methods\.[A-Za-z_][A-Za-z_0-9]*',pair[0]) and pair[1].isidentifier(),'Unsafe extension name')
        require(pair not in registry,'Duplicate extension registry pair')
        expected='code/'+pair[0].replace('.','/')+'.py'
        require(expected in item['source_files'],'Extension must lock its selected module')
        for name,digest in item['source_files'].items():
            p=(ROOT/name).resolve();require(p.is_relative_to(ROOT.resolve()),'Extension source escapes project')
            require(sha(p)==digest,'Extension source identity differs: '+name)
        receipt=resolve_path(item['support_receipt'],DATA_ROOT)
        require(sha(receipt)==item['support_receipt_sha256'],'Unverified extension support receipt')
        registry[pair]=deepcopy(item)
    return registry


def _extension_plan(path,registry):
    """Common validation for an explicitly registered future timing-only port."""
    from dataset_registry import load_inputs
    from model_registry import get_model_spec
    cfg=read(path);spec=get_model_spec(cfg['model_id'])
    require(not set(cfg)&{'refs','references','reference_answers','reference_path','answers','labels','scores'},'Reference fields forbidden')
    require(cfg.get('model_revision',spec['revision'])==spec['revision'],'Checkpoint revision differs')
    pair=(cfg['method'].get('adapter_module'),cfg['method'].get('adapter_class'))
    require(pair in registry,'Unapproved timing extension')
    require(cfg['method'].get('information_condition','query_unknown')=='query_unknown','Query-known requires a separate protocol')
    cfg.setdefault('question_suffix',DEFAULT_SUFFIX)
    rows=load_inputs(cfg['inputs'])
    require(rows and len({r['dataset'] for r in rows})==1,'One nonempty dataset per timing job')
    return cfg,spec,rows


def select_rows(job,rows,base_config=None):
    lookup={row['id']:row for row in rows}
    if job.get('ids') is not None:
        ids=job['ids'];require(isinstance(ids,list) and ids and len(ids)==len(set(ids)),'Invalid fixed timing IDs')
        return [lookup[identity] for identity in ids],{'mode':'explicit_fixed_ids','labels_or_scores_used':False}
    selection=job.get('selection',{'mode':'first','count':16})
    mode=selection.get('mode','first');count=int(selection.get('count',16))
    require(0<count<=len(rows),'Invalid timing QA count')
    if mode=='first':return rows[:count],{'mode':'first_in_frozen_input_order','count':count,'labels_or_scores_used':False}
    require(mode=='native_visual_tokens_quantile','Unknown timing selection mode')
    # This optional selection consumes only N/id/image identity from prior
    # native records; no reference file or quality score is read.
    reference=Path(selection['native_reference_records']);counts={}
    reference_manifest=read(reference.parent/'manifest.json')
    require(reference_manifest['config']['method']['family'].lower() in ('fp16','native'),'N strata require a native FP16 reference')
    if base_config is not None:
        for key in ('model_id','model_revision','preprocessing'):
            require(reference_manifest['config'].get(key)==base_config.get(key),'N strata reference differs in '+key)
    with reference.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('status')=='PASS' and row['id'] in lookup:
                require(row['original_image_sha256']==lookup[row['id']]['original_image_sha256'],'N reference image mismatch')
                counts[row['id']]=int(row['actual_visual_token_count'])
    require(set(counts)==set(lookup),'Native N reference must cover the entire fixed timing candidate cohort')
    ranked=sorted(rows,key=lambda row:(counts[row['id']],hashlib.sha256((str(job.get('seed',20261120))+'\0'+row['id']).encode()).hexdigest()))
    chosen=[ranked[min(len(ranked)-1,int((i+.5)*len(ranked)/count))] for i in range(count)]
    return chosen,{'mode':mode,'count':count,'native_reference_sha256':sha(reference),'native_reference_manifest_sha256':sha(reference.parent/'manifest.json'),
                   'selection_fields':['id','status','original_image_sha256','actual_visual_token_count'],
                   'selected_N':[counts[r['id']] for r in chosen],'labels_or_scores_used':False}


def make_plan(job_path,output):
    """Create immutable effective configs and fixed schedule without a model."""
    import evaluate
    from dataset_registry import load_inputs
    job=read(job_path)
    require(not(set(job)-JOB_FIELDS),'Unknown timing job field: '+repr(sorted(set(job)-JOB_FIELDS)))
    require(job.get('schema')=='baseline_matrix.speed_job.v1','Explicit timing schema required')
    require(len(job.get('eval_configs',[]))>=2,'Interleaved speed comparison needs at least two methods')
    repeat=int(job.get('repeat',3));warm=int(job.get('warmup_per_qa',1));seed=int(job.get('seed',20261120))
    require(repeat>=3 and warm>=1,'Use at least three measured repeats and one full warmup per QA/method')
    require(job.get('allocator_policy','retain_caching_allocator') in ('retain_caching_allocator','empty_between_trials'),'Unknown allocator policy')
    registry=extension_registry(job)
    base_paths=[resolve_path(value,ROOT) for value in job['eval_configs']]
    original=[read(p) for p in base_paths]
    common_input=resolve_path(job.get('inputs',original[0]['inputs']),ROOT)
    require(common_input.name.endswith('_inputs.jsonl'),'Timing inputs must be reference-free *_inputs.jsonl')
    rows=load_inputs(common_input);chosen,selection=select_rows(job,rows,original[0])
    require(len({r['dataset'] for r in chosen})==1,'One dataset per timing benchmark')
    lockfile=common_input.parent/'source_lock.json'
    require(lockfile.exists(),'Timing requires the frozen dataset source lock')
    lock=read(lockfile)['files'][common_input.name]
    require(sha(common_input)==lock['sha256'] and common_input.stat().st_size==lock['bytes'],'Timing cohort SHA differs')
    configs=[];specs=[]
    for index,(base,path) in enumerate(zip(original,base_paths)):
        cfg=deepcopy(base);cfg.update(inputs=str(common_input),ids=[row['id'] for row in chosen],start=0,count=len(chosen),
             scope='FIXED_SCORE_BLIND_SPEED_COHORT; native whole-request replay, no reference answers')
        file=output/'effective_eval_configs'/f'{index:02d}.json';write_new(file,cfg)
        pair=(cfg['method'].get('adapter_module'),cfg['method'].get('adapter_class'))
        if pair in registry and pair not in evaluate.ALLOWLIST:resolved,spec,_=_extension_plan(file,registry)
        else:resolved,spec,_=evaluate.load_plan(file)
        configs.append(resolved);specs.append(spec)
    compare_fields=('model_id','model_revision','model_path','model_source_lock','preprocessing','question_suffix','max_new_tokens')
    for cfg,spec in zip(configs[1:],specs[1:]):
        require(spec==specs[0],'Load exactly one checkpoint/model per timing job')
        for key in compare_fields:require(cfg.get(key)==configs[0].get(key),'Methods differ in common inference behavior: '+key)
    require(all(cfg['method']['family'].lower()!='full_native' for cfg in configs),'Main speed anchor is native image-prefix plus suffix, not independent full-native generation')
    methods=[cfg['method']['id'] for cfg in configs]
    require(len(methods)==len(set(methods)),'Duplicate method IDs')
    rng=random.Random(seed);schedule=[]
    for phase,nrepeat in (('warmup',warm),('measure',repeat)):
        for round_index in range(nrepeat):
            for row in chosen:
                order=list(range(len(methods)));rng.shuffle(order)
                for order_index,index in enumerate(order):
                    schedule.append({'phase':phase,'repeat':round_index,'id':row['id'],'method':methods[index],
                                     'config_index':index,'method_order_within_QA':order_index})
    plan={'schema':'baseline_matrix.speed_plan.v1','status':'CPU_PLAN_ONLY_NO_GPU',
          'job':job,'job_sha256':sha(job_path),'model_spec':specs[0],
          'effective_configs':configs,'original_config_hashes':{str(p):sha(p) for p in base_paths},
          'inputs_sha256':sha(common_input),'source_lock_sha256':sha(lockfile),
          'expected_ids':[r['id'] for r in chosen],'selection':selection,
          'method_ids':methods,'measured_trials':len(chosen)*len(methods)*repeat,
          'warmup_trials':len(chosen)*len(methods)*warm,'schedule':schedule,
          'references_read':False,'GPU_used':False,'source_sha256':sha(Path(__file__)),
          'extensions':list(registry.values())}
    write_new(output/'plan.json',plan)
    return plan,chosen,registry


def expected_predictions(path,plan):
    if path is None:return {},None
    value=read(path)
    require(value.get('schema')=='baseline_matrix.expected_model_predictions.v1','Expected file must be model outputs, never task references')
    require(value.get('references_read') is False,'Expected outputs must not consume task references')
    from prediction_cache import normalize_behavior
    require(set(plan['method_ids']).issubset(value.get('source_contracts_by_method',{})),'Expected outputs require verified quality source contexts by method')
    for cfg in plan['effective_configs']:
        context=value['source_contracts_by_method'][cfg['method']['id']]
        require(normalize_behavior(cfg,plan['model_spec'])==context['behavior'],'Expected quality/timing config behavior differs')
    lookup={}
    for row in value['records']:
        require(not set(row)&{'answers','labels','ground_truth','reference_answers','official_score','score'},'Ground truth/scoring fields forbidden in expected model outputs')
        key=(row['method'],row['id']);require(key not in lookup,'Duplicate expected model output')
        require(isinstance(row['output_token_ids'],list) and row['output_token_ids'],'Expected genuine generated token IDs')
        require(row['model_id']==plan['model_spec']['model_id'] and row['model_commit']==plan['model_spec']['revision'],'Expected checkpoint differs')
        lookup[key]=row
    required={(method,identity) for method in plan['method_ids'] for identity in plan['expected_ids']}
    require(required.issubset(lookup),'Expected file must cover all timing methods/IDs')
    return lookup,sha(path)


def verify_expected_contexts(path,plan,identity,model,torch,transformers):
    """Same quality configuration/runtime/source, beyond coincident answer IDs."""
    if path is None:return {'status':'NO_EXPECTED_QUALITY_CONTEXT'}
    from prediction_cache import evaluate_semantics
    value=read(path);files=sorted(({k:r[k] for k in ('file','bytes','sha256')} for r in identity['files']),key=lambda r:r['file'])
    actual_runtime={'torch_version':torch.__version__,'transformers_version':transformers.__version__,
                    'cuda_version':torch.version.cuda,'python_version':__import__('platform').python_version(),
                    'hardware':torch.cuda.get_device_name(),'runtime_executable':sys.executable}
    for method in plan['method_ids']:
        context=value['source_contracts_by_method'][method]
        require(context['model_files']==files,'Quality/timing checkpoint or processor bytes differ')
        require(context['runtime']==actual_runtime,'Quality/timing runtime differs')
        require(context['generation_native']==model.generation_config.to_dict(),'Quality/timing native generation differs')
        for relative,digest in context['relevant_source_hashes'].items():
            path=ROOT/relative
            actual=evaluate_semantics(path,sha(path)) if relative=='code/evaluate.py' else sha(path)
            require(actual==digest,'Quality/timing source differs: '+relative)
    return {'status':'PASS_QUALITY_RUNTIME_SOURCE_CHECKPOINT_CONTEXT','methods':len(plan['method_ids'])}


class ArrivalObserver:
    """Natural CPU token arrival; a single suffix CUDA interval, no finite hook."""
    def __init__(self,model,batch,torch):
        self.model,self.batch,self.torch=model,batch,torch
        self.phase='prefix';self.prompt_seen=False;self.ids=[];self.times=[];self.chunks=[]
        self.first_started=False;self.first_finished=False
        self.first_start=torch.cuda.Event(enable_timing=True);self.first_end=torch.cuda.Event(enable_timing=True)
        self.handles=[]
    def put(self,value):
        now=time.perf_counter()
        if not self.prompt_seen:
            self.prompt_seen=True;self.prompt_chunk_length=value.numel();return
        tokens=value.detach().cpu().reshape(-1).tolist()
        self.ids.extend(tokens);self.times.extend([now]*len(tokens));self.chunks.append(len(tokens))
    def end(self):pass
    def before(self,module,args,kwargs):
        if self.phase!='generation' or self.first_started:return
        # Native qwen_generate_cached activates a new tiny delta before the
        # first forward. Drop the obsolete batch reference just as quality does.
        if hasattr(self.batch,'rope_deltas'):self.batch.rope_deltas=self.model.model.rope_deltas
        self.first_started=True;self.first_host_start=time.perf_counter();self.first_start.record()
    def after(self,module,args,kwargs,output):
        if self.phase!='generation' or self.first_finished:return
        self.first_end.record();self.first_host_end=time.perf_counter();self.first_finished=True
        for handle in self.handles:handle.remove()


@contextmanager
def stream_native_generation(model,observer):
    original=model.generate
    observer.handles=[model.register_forward_pre_hook(observer.before,with_kwargs=True),
                      model.register_forward_hook(observer.after,with_kwargs=True)]
    def generate(*args,**kwargs):return original(*args,streamer=observer,return_dict_in_generate=False,**kwargs)
    model.generate=generate
    try:yield
    finally:
        model.generate=original
        for handle in observer.handles:handle.remove()


def exclusive_gpu_check():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(re.fullmatch(r'[0-9]+',visible) is not None,'Declare exactly one physical GPU index in CUDA_VISIBLE_DEVICES')
    result=subprocess.run(['nvidia-smi','-i',visible,'--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    processes=[int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    require(all(pid==os.getpid() for pid in processes),'EXCLUSIVE_GPU_CHECK_FAILED: another compute process is present')
    return {'physical_GPU_index':visible,'observed_compute_PIDs':processes,'own_PID':os.getpid()}


def make_adapter(method,index,registry):
    import evaluate
    pair=(method['adapter_module'],method['adapter_class'])
    if pair in evaluate.ALLOWLIST:return evaluate.adapter_for(method,index)
    require(pair in registry,'Timing adapter extension not explicitly registered')
    cls=getattr(importlib.import_module(pair[0]),pair[1]);args=dict(method.get('constructor_args',{}))
    if method.get('per_layer_seed'):
        seed=method['per_layer_seed'];args['seed']=int(seed['base'])+int(seed['step'])*index
    return cls(**args)


def run_trial(model,processor,protocol,prepare,row,image_bytes,cfg,spec,item,registry,expected):
    import torch
    from PIL import Image
    from cache_bridge import HybridVisualCache
    from attention_bridge import attention_context
    from ownership import cache_storage
    from evaluate import tensor_sha
    method=cfg['method'];family=method['family'].lower()
    cache=batch=observer=capture=owners=generated=image=prefix_output=None
    record={**item,'model_id':spec['model_id'],'model_commit':spec['revision'],'dataset':row['dataset'],
            'dataset_revision':row['dataset_revision'],'original_image_sha256':row['original_image_sha256'],
            'method_contract':method,'status':'RUNNING','canonical_speed_eligible':False,
            'reference_answers_read':False,'timing_scope':'streaming complete request from RAM-resident original image bytes; disk I/O excluded'}
    events={name:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for name in ('preprocess','prefill','encode','generation')}
    host={}
    try:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        record.update(start_allocated_bytes=torch.cuda.memory_allocated(),start_reserved_bytes=torch.cuda.memory_reserved())
        started=time.perf_counter();events['preprocess'][0].record();t=started
        with Image.open(io.BytesIO(image_bytes)) as raw:image=raw.convert('RGB')
        batch=prepare(image,row['question']+cfg['question_suffix'])
        host['preprocess']=(time.perf_counter()-t)*1000;events['preprocess'][1].record()
        observer=ArrivalObserver(model,batch,torch)
        with stream_native_generation(model,observer):
            events['prefill'][0].record();t=time.perf_counter()
            capture_enabled=method.get('encode_scope','visual')=='prefix'
            window=int(method.get('constructor_args',{}).get('window_size',64))
            context=attention_context(model,capture_queries=True,window_size=window) if capture_enabled else nullcontext(None)
            with context as capture:prefix_output=protocol['prefix'](model,batch)
            cache=prefix_output.past_key_values;prefix_output=None
            host['prefill']=(time.perf_counter()-t)*1000;events['prefill'][1].record()
            events['encode'][0].record();t=time.perf_counter();vp=batch.visual_positions
            if hasattr(batch,'rope_deltas'):batch.rope_deltas=model.model.rope_deltas
            if family not in ('fp16','native'):
                owners=None
                if family!='fp16_hybrid':
                    owners=[]
                    for index,layer in enumerate(cache.layers):
                        adapter=make_adapter(method,index,registry)
                        if capture_enabled:
                            owner=adapter.encode(layer.keys,layer.values,visual_positions=vp,
                                window_queries=capture['queries_by_layer'][index],layer_idx=index,
                                prefix_is_question_free=True,question_in_observation=False)
                        else:owner=adapter.encode(layer.keys.index_select(-2,vp),layer.values.index_select(-2,vp),positions=vp,layer_idx=index)
                        owners.append(owner)
                    del layer,owner,adapter
                cache=HybridVisualCache(cache,vp,owners,reader_backend=method.get('reader_backend'),consume_native=True)
                owners=None
            capture=None
            # Preserve the quality protocol's native delta and consume image
            # tensors/maps before suffix replay; no owner reports or GC here.
            keep_ids,keep_mask=batch.ids,batch.mask
            batch.inputs={'input_ids':keep_ids,'attention_mask':keep_mask};batch.prefix_inputs={}
            batch.positions=batch.positions.detach().cpu();batch.visual_positions=batch.visual_positions.detach().cpu()
            host['encode']=(time.perf_counter()-t)*1000;events['encode'][1].record()
            observer.phase='generation';events['generation'][0].record();t=time.perf_counter()
            context=attention_context(model,cache=cache) if hasattr(cache,'named_tensors') else nullcontext()
            with context:generated=protocol['generate'](model,batch,cache,max_new_tokens=int(cfg.get('max_new_tokens',64)))
            # Text decoding is real request work; comparison/hash/owner audit is
            # not. The stream already holds the generated CPU token sequence.
            prediction=processor.decode(observer.ids,skip_special_tokens=True)
            host['generation']=(time.perf_counter()-t)*1000;events['generation'][1].record()
            torch.cuda.synchronize();finished=time.perf_counter();generation_wall_ms=(finished-t)*1000
        # End of the timed request. Everything below is evidence/validation.
        record.update(wall_ms=(finished-started)*1000,preprocess_ms=host['preprocess'],
            prefill_ms=events['prefill'][0].elapsed_time(events['prefill'][1]),
            encode_ms=events['encode'][0].elapsed_time(events['encode'][1]),
            generation_ms=events['generation'][0].elapsed_time(events['generation'][1]),
            generation_wall_ms=generation_wall_ms,
            phase_host_dispatch_ms=host,
            phase_GPU_stream_interval_ms={key:a.elapsed_time(b) for key,(a,b) in events.items()},
            question_prefill_ms=observer.first_start.elapsed_time(observer.first_end) if observer.first_finished else None,
            question_prefill_host_dispatch_ms=(observer.first_host_end-observer.first_host_start)*1000 if observer.first_finished else None,
            ttft_ms=(observer.times[0]-started)*1000 if observer.times else None,
            inter_token_ms=[(b-a)*1000 for a,b in zip(observer.times,observer.times[1:])],
            output_chunk_sizes=observer.chunks,raw_prediction=prediction,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        tokens=generated[0,batch.full_length:].detach().cpu().tolist()
        require(tokens and tokens==observer.ids,'Native returned IDs differ from streamed generated IDs')
        require(observer.prompt_chunk_length==batch.full_length,'Unexpected initial streamer prompt shape')
        record.update(output_token_ids=tokens,output_tokens=len(tokens),
            generation_tokens_per_second=len(tokens)/(generation_wall_ms/1000) if generation_wall_ms else None,
            full_request_tokens_per_second=len(tokens)/((finished-started)) if finished>started else None,
            mean_inter_token_ms=statistics.mean(record['inter_token_ms']) if record['inter_token_ms'] else None,
            decode_tokens_per_second=(len(tokens)-1)/(sum(record['inter_token_ms'])/1000) if len(tokens)>1 and sum(record['inter_token_ms'])>0 else None,
            input_ids_sha256=tensor_sha(batch.ids),native_position_ids_sha256=tensor_sha(batch.positions),
            full_prompt_hash=hashlib.sha256(batch.prompt.encode()).hexdigest(),
            actual_visual_token_count=int(batch.visual_positions.numel()),status='PASS')
        common={'rope_deltas':model.model.rope_deltas} if hasattr(batch,'rope_deltas') else {}
        report=cache_storage(cache,batch.visual_positions,common_tensors=common)
        denom=spec['layers']*2*spec['kv_heads']*record['actual_visual_token_count']*spec['head_dim']*2
        record.update(complete_visual_owner_bytes=report['complete_visual_owner_bytes'],stored_visual_KV_bytes=report['stored_visual_KV_bytes'],
            actual_effective_visual_bits=16*report['complete_visual_owner_bytes']/denom,full_kv_byte_denominator=denom,
            owner_after_timing=report,owner_audit_in_timed_region=False)
        target=expected.get((method['id'],row['id']))
        if target is None:record['quality_token_identity_status']='NOT_CHECKED_NO_EXPECTED_OUTPUT'
        else:
            same=tokens==target['output_token_ids']
            for key in ('original_image_sha256','input_ids_sha256','full_prompt_hash'):
                if key in target:require(record[key]==target[key],'Quality/timing input mismatch: '+key)
            record['quality_token_identity_status']='EXACT_MATCH' if same else 'MISMATCH'
            record['canonical_speed_eligible']=same
            if target.get('stored_visual_KV_bytes') is not None:
                require(record['stored_visual_KV_bytes']==target['stored_visual_KV_bytes'],'Quality/timing complete visual owner bytes differ')
            if not same:record['status']='TOKEN_ID_MISMATCH_WITH_QUALITY'
    except Exception as exc:
        record.update(status='OOM_AT_DECLARED_INPUT' if 'out of memory' in str(exc).lower() else 'FAILED_RUN',
                      error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc(),canonical_speed_eligible=False)
    finally:
        cache=batch=observer=capture=owners=generated=image=prefix_output=None
    return record


def summarize(records,plan):
    groups=defaultdict(list)
    for row in records:
        if row['phase']=='measure':groups[row['method']].append(row)
    methods={}
    for method in plan['method_ids']:
        rows=groups[method];passed=[r for r in rows if r['status']=='PASS']
        warmup=[r for r in records if r['phase']=='warmup' and r['method']==method]
        warmup_ok=len(warmup)==len(plan['expected_ids'])*int(plan['job'].get('warmup_per_qa',1)) and all(r['status']=='PASS' and r['canonical_speed_eligible'] for r in warmup)
        eligible=warmup_ok and len(rows)==len(plan['expected_ids'])*int(plan['job'].get('repeat',3)) and all(r['canonical_speed_eligible'] for r in rows)
        item={'expected_trials':len(plan['expected_ids'])*int(plan['job'].get('repeat',3)),
              'recorded_trials':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
              'warmup_status_counts':dict(Counter(r['status'] for r in warmup)),
              'canonical_speed_eligible':eligible,'statistics_scope':'all successful trials; incomplete groups never become canonical',
              'primary_mean_wall_ms':statistics.mean([r['wall_ms'] for r in rows]) if eligible else None,
              'no_wall_only_ranking_when_output_lengths_differ':True}
        if passed:
            for key in ('wall_ms','ttft_ms','question_prefill_ms','encode_ms','output_tokens','mean_inter_token_ms',
                        'full_request_tokens_per_second','generation_tokens_per_second','decode_tokens_per_second','peak_allocated_bytes','complete_visual_owner_bytes'):
                values=[r[key] for r in passed if r.get(key) is not None]
                item[key]={'mean':statistics.mean(values),'median':statistics.median(values),'n':len(values)} if values else None
        methods[method]=item
    return {'schema':'baseline_matrix.speed_summary.v1','status':'PASS_TIMED_AND_TOKEN_VALIDATED' if all(m['canonical_speed_eligible'] for m in methods.values()) else 'COMPLETE_WITH_UNVERIFIED_OR_FAILED_TIMING',
            'methods':methods,'quality_scores_read':False,'timing_is_streaming_E2E':True,
            'phase_times':'CUDA stream intervals without per-phase synchronization; host dispatch separately; wall ends after final synchronization',
            'source_qa_pairing_required_for_claims':True,'output_length_adjusted_analysis_required':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan,rows,registry=make_plan(args.config,args.output)
    expected_path=plan['job'].get('expected_predictions')
    expected,expected_sha=expected_predictions(resolve_path(expected_path,ROOT) if expected_path else None,plan)
    if args.plan:
        write_new(args.output/'complete.json',{'status':'PASS_CPU_SPEED_PLAN','GPU_used':False,
            'measured_trials':plan['measured_trials'],'warmup_trials':plan['warmup_trials'],
            'expected_model_output_file_sha256':expected_sha,'timing_numbers_produced':False})
        print(json.dumps({'status':'PASS_CPU_SPEED_PLAN','measured_trials':plan['measured_trials'],'warmup_trials':plan['warmup_trials']}));return
    import torch
    import transformers
    from PIL import __version__ as pillow_version
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from ownership import model_storage
    from evaluate import model_identity
    torch.set_grad_enabled(False);torch.manual_seed(int(plan['job'].get('seed',20261120)))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one visible CUDA GPU is required')
    torch.cuda.set_device(0);require(torch.cuda.get_device_capability()==(8,6),'Declared RTX3090 SM86 required')
    exclusive=exclusive_gpu_check() if plan['job'].get('verify_gpu_exclusive',True) else {'status':'ROOT_DISABLED_EXCLUSIVITY_CHECK'}
    cfg=plan['effective_configs'][0];spec=plan['model_spec'];identity=model_identity(cfg,spec)
    processor=AutoProcessor.from_pretrained(cfg['model_path'],local_files_only=True)
    t=time.perf_counter();model=AutoModelForImageTextToText.from_pretrained(cfg['model_path'],dtype=torch.float16,
        device_map={'':0},local_files_only=True,attn_implementation='sdpa').eval()
    model.requires_grad_(False);torch.cuda.synchronize();load_ms=(time.perf_counter()-t)*1000
    require(all(p.dtype==torch.float16 for p in model.parameters()) and not getattr(model,'is_quantized',False),'Original FP16 weights required')
    require(int(model.generation_config.num_beams or 1)==1,'Only native greedy beam-one generation supported')
    context_validation=verify_expected_contexts(resolve_path(expected_path,ROOT) if expected_path else None,plan,identity,model,torch,transformers)
    module=importlib.import_module('m1_protocol' if spec['model_id']=='M1' else 'qwen_protocol')
    if spec['model_id']=='M1':
        prepare=lambda image,question:module.prepare_m1_inputs(processor,image,question,image_token_id=spec['image_token_id'],max_patches=int(cfg['preprocessing'].get('max_patches',8)))
        protocol={'prefix':module.m1_prefix_forward,'generate':module.m1_generate_cached}
    else:
        prepare=lambda image,question:module.prepare_qwen_inputs(processor,image,question,model=model,image_token_id=spec['image_token_id'],**cfg['preprocessing'])
        protocol={'prefix':module.prefix_forward,'generate':module.generate_cached}
    blobs={}
    preflight=[]
    for row in rows:
        body=Path(row['image_path']).read_bytes();require(hashlib.sha256(body).hexdigest()==row['original_image_sha256'],'Original image SHA mismatch')
        blobs[row['id']]=body
        # Question independence is a protocol audit, performed once per QA
        # before warmup/timing rather than doubled inside every measured call.
        from PIL import Image
        with Image.open(io.BytesIO(body)) as raw:image=raw.convert('RGB')
        batch=prepare(image,row['question']+cfg['question_suffix'])
        alternate=prepare(image,'What is visible in the image?'+cfg['question_suffix'])
        require(batch.boundary==alternate.boundary and torch.equal(batch.ids[:,:batch.boundary],alternate.ids[:,:alternate.boundary]),'Question-dependent image prefix')
        for key in ('pixel_values','image_grid_thw','pixel_attention_mask'):
            if key in batch.inputs:require(torch.equal(batch.inputs[key],alternate.inputs[key]),'Question-dependent vision processing')
        preflight.append({'id':row['id'],'actual_visual_token_count':int(batch.visual_positions.numel()),'question_unknown_prefix_verified':True})
        del batch,alternate,image
    manifest={'schema':'baseline_matrix.speed_actual_run.v1','plan_sha256':sha(args.output/'plan.json'),
        'model_identity':identity,'model_load_ms_outside_timing':load_ms,'model_residency':model_storage(model),
        'generation_native':model.generation_config.to_dict(),'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
        'cuda':torch.version.cuda,'pillow':pillow_version,'hardware':torch.cuda.get_device_name(),'visible_GPU':os.environ.get('CUDA_VISIBLE_DEVICES')},
        'initial_exclusive_GPU_check':exclusive,'host_quiet_asserted':plan['job'].get('host_quiet_asserted',False),
        'expected_predictions_sha256':expected_sha,'references_read':False,'model_loads':1,
        'quality_context_validation':context_validation,'preflight_question_independence':preflight,
        'input_disk_read_scope':'original encoded image bytes read and SHA-verified before timing, decoded anew inside every request',
        'allocator_policy':plan['job'].get('allocator_policy','retain_caching_allocator'),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/evaluate.py',ROOT/'code/ownership.py',ROOT/'code/cache_bridge.py',ROOT/'code/attention_bridge.py',Path(module.__file__)]},
        'adapter_source_hashes':{p.name:sha(p) for p in (ROOT/'code/methods').glob('*.py')}}
    write_new(args.output/'manifest.json',manifest)
    lookup={r['id']:r for r in rows};records=[];last_qa=None
    for item in plan['schedule']:
        if plan['job'].get('verify_gpu_exclusive',True) and item['id']!=last_qa:exclusive_gpu_check()
        last_qa=item['id'];gc.collect()
        if plan['job'].get('allocator_policy','retain_caching_allocator')=='empty_between_trials':torch.cuda.empty_cache()
        torch.manual_seed(int(plan['job'].get('seed',20261120)))
        record=run_trial(model,processor,protocol,prepare,lookup[item['id']],blobs[item['id']],plan['effective_configs'][item['config_index']],spec,item,registry,expected)
        records.append(record)
        with (args.output/'records.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({k:record[k] for k in ('phase','repeat','id','method','status')}),flush=True)
    summary=summarize(records,plan);write_new(args.output/'summary.json',summary)
    write_new(args.output/'complete.json',{'status':summary['status'],'recorded_trials':len(records),'expected_trials':len(plan['schedule']),
        'warmup_and_measure_status_counts':dict(Counter(r['status'] for r in records)),
        'manifest_sha256':sha(args.output/'manifest.json'),'records_jsonl_sha256':sha(args.output/'records.jsonl'),
        'summary_sha256':sha(args.output/'summary.json'),'GPU_used':True})
    print(json.dumps({'status':summary['status'],'recorded_trials':len(records)}))


if __name__=='__main__':main()
