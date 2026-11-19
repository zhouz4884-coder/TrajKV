"""Shared CPU preflight/source/postcheck support for new backend wrappers.

Only explicitly allowed backend modules/families are imported. Numerical
classes and original diagnostic/timer functions are never rewritten.
"""
from contextlib import contextmanager
import importlib
import inspect
import json
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parents[2]
ALLOWED={'round8_candidate_runner':{'spherical','field'},'round8_postres_runner':{'postres'},
         'round8_final_ray_runner':{'final_ray'}}
CORE_NAMES=('round8_runner.py','round8_bank.py','round8_codec.py','round8_attention.py','round8_joint_gain.py',
    'round7_runner.py','round6_runner.py','session_boundary.py','factory.py','masked_bank.py',
    'metadata_direction_bank.py','round6_depth_bank.py','round6_depth_attention.py','round6_depth_codec.py',
    'round7_key_bank.py','round7_key_attention.py','round7_key_codec.py','round7_key_layout.py','round7_key_artifacts.py',
    'hybrid_packed_bank.py','fused_hybrid.py','fast_wht.py','quantizers.py','affine_rate_bank.py')


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def resolve_backend(config):
    name,family=config.get('backend_module'),config.get('backend_family')
    if name not in ALLOWED or family not in ALLOWED[name]:
        raise ValueError('Explicit backend_module/family is not one of the verified allowed pairs')
    module=importlib.import_module(name)
    return module,family


def backend_sources(module,family):
    if module.__name__=='round8_candidate_runner':
        with module.scoped_backend(family) as selected:return dict(selected.source_fingerprints())
    return dict(module.source_fingerprints())


def validate_backend_config(config):
    module,family=resolve_backend(config)
    if not config.get('variants'):raise ValueError('Explicit variants required')
    caps={int(v.get('cap',8)) for v in config['variants']}
    if caps!={8} or int(config['arguments'].get('max_patches',8))!=8:
        raise ValueError('Backend diagnostic/quiet wrappers preserve common cap8')
    for variant in config['variants']:
        if variant.get('backend_family',family)!=family or variant.get('backend_module',module.__name__)!=module.__name__:
            raise ValueError('Cross-family/module variants are not supported')
        if family!='field' and variant.get('value_joint_refit',False):raise ValueError('Selected backend would ignore V refit')
        if family!='spherical' and variant.get('key_assignment')=='row_cosine_cd':raise ValueError('Selected backend would ignore CD')
        if family!='postres' and variant.get('postres_direction',False):raise ValueError('Selected backend would ignore postres direction')
        if family!='final_ray' and variant.get('final_ray_mode','off')!='off':raise ValueError('Selected backend would ignore final-state ray mode')
    if module.__name__=='round8_candidate_runner':
        declarations=module.validate_subset_config(config,family,module.load_backend(family))
    else:declarations=module.validate_config(config)
    return module,family,declarations


def source_path(name):
    name=str(name)
    if name.startswith('code/'):return ROOT/name
    if name.startswith('experiments/'):return ROOT/'code'/name
    if '/' in name or '\\' in name:raise ValueError('Unexpected backend source-index key: '+name)
    return Path(__file__).with_name(name)


def source_snapshot(module,family,wrapper_path,engine_name):
    files={Path(__file__),Path(wrapper_path),Path(__file__).with_name(engine_name)}
    files.update(Path(__file__).with_name(name) for name in CORE_NAMES)
    files.update(ROOT/name for name in ('code/experiments/bank.py','code/experiments/timing_control.py','code/bytes.py'))
    indexed=backend_sources(module,family)
    files.add(Path(module.__file__))
    for name,expected in indexed.items():
        path=source_path(name)
        if sha(path)!=expected:raise RuntimeError('Backend fingerprint differs from actual source: '+name)
        files.add(path)
    return {str(path.resolve()):sha(path) for path in sorted(files)}


def selected_rows(config,engine):
    rows=read(config['arguments']['samples'])
    if engine=='round7_timing.py':return rows[:3]
    original=importlib.import_module('round8_continuous')
    return original.select_sources(rows,int(config.get('count',8)),bool(config.get('balance_datasets',True)))


def input_snapshot(config_path,config,engine,reference_arg=None,strict=True):
    roles={}
    def add(value,role):
        if value is None:return
        path=Path(value)
        roles.setdefault(str(path.resolve()),set()).add(role)
    add(config_path,'wrapper/original config');args=config['arguments']
    add(args.get('samples'),'answer-free source/QA metadata')
    if args.get('model'):add(Path(args['model'])/'config.json','model config only; weights not rehashed')
    if args.get('kivi_source'):add(args['kivi_source'],'upstream packed-code source')
    references=reference_arg or config.get('diagnostic_references')
    add(references,'optional canonical-GT diagnostic references')
    for variant in config['variants']:
        for key,value in variant.items():
            if key.endswith('_path') and isinstance(value,str) and value:add(value,'variant input '+key)
            elif key.endswith('_paths') and isinstance(value,list):
                for item in value:add(item,'variant inputs '+key)
    contract=config.get('frozen_variant_contract')
    if contract:
        add(contract['path'],'exact frozen quality-variant contract')
        if Path(contract['path']).is_file() and sha(contract['path'])!=contract['sha256']:
            raise ValueError('Frozen quality-variant contract hash differs')
        if Path(contract['path']).is_file():
            frozen={v['name']:v for v in read(contract['path'])['variants']}
            if any(frozen.get(v['name'])!=v for v in config['variants']):
                raise ValueError('Variant differs from the frozen quality contract')
    legacy=Path(args['legacy_scripts']) if args.get('legacy_scripts') else None
    if legacy and legacy.is_dir():
        for path in legacy.rglob('*.py'):
            if '__pycache__' not in path.parts:add(path,'legacy runtime source')
    sample_path=Path(args['samples'])
    if sample_path.is_file():
        for row in selected_rows(config,engine):
            for path in (row.get('remote_images') or row.get('images') or []):add(path,'actual selected image bytes')
    result,missing,directories={},[],{}
    if legacy:
        if not legacy.is_dir():missing.append(dict(path=str(legacy.resolve()),roles=['legacy runtime source directory'],expected_type='directory'))
        else:directories[str(legacy.resolve())]=dict(python_source_files=sorted(str(p.resolve()) for p in legacy.rglob('*.py') if '__pycache__' not in p.parts))
    for name,labels in sorted(roles.items()):
        path=Path(name)
        if not path.is_file():
            missing.append(dict(path=name,roles=sorted(labels)))
            continue
        result[name]=dict(sha256=sha(path),roles=sorted(labels))
    if strict and missing:raise FileNotFoundError('Required wrapper fingerprint inputs unavailable: '+', '.join(x['path'] for x in missing))
    return dict(files=result,directories=directories,missing=missing)


@contextmanager
def registered_backend(module,family,variants):
    # Cache the whole original R8/R7/R6 registration chain BEFORE the patch.
    importlib.import_module('round8_runner')
    with module.backend_registration(family) as cls:
        factory=importlib.import_module('factory')
        executor=importlib.import_module('run_representation_variants')
        session=importlib.import_module('session_boundary')
        if factory.bank_class('round8_joint') is not cls or executor.bank_class('round8_joint') is not cls:
            raise RuntimeError('Chosen backend is not the actual registered round8_joint class')
        round8_bank=importlib.import_module('round8_bank')
        if cls is round8_bank.Round8Bank:raise RuntimeError('Default Round8Bank would silently ignore new fields')
        # Both original tools also run an internal Full teacher/canonical pass.
        classes={factory.bank_class(v['method']) for v in variants}|{factory.bank_class('full')}
        states=[(item,'settings' in item.__dict__,getattr(item,'settings',None)) for item in classes]
        previous_bank=session.ProgressiveBank
        source=inspect.getsourcefile(cls)
        identity=dict(module=cls.__module__,qualname=cls.__qualname__,source_path=source,
            source_sha256=sha(source) if source and Path(source).is_file() else None,
            mro=[item.__module__+'.'+item.__qualname__ for item in cls.__mro__],
            factory_identity_verified=True,executor_identity_verified=True,original_R8_registration_preimported=True)
        try:yield cls,identity
        finally:
            session.ProgressiveBank=previous_bank
            for item,owned,settings in states:
                if owned:item.settings=settings
                elif 'settings' in item.__dict__:delattr(item,'settings')


@contextmanager
def supported_round8(timer):
    previous=timer.SUPPORTED
    timer.SUPPORTED=set(previous)|{'round8_joint'}
    try:yield
    finally:timer.SUPPORTED=previous


def artifact_snapshot(output):
    return {path.relative_to(output).as_posix():sha(path) for path in sorted(Path(output).rglob('*')) if path.is_file()}


def complete_wrapper(output,kind,module,family,declarations,before,after,class_identity,wrapper_path):
    output=Path(output)
    if before!=after:raise RuntimeError('Backend wrapper source/input changed; preserve run and use a new output')
    original=artifact_snapshot(output)
    if 'complete.json' not in original or 'manifest.json' not in original:raise RuntimeError('Original engine lacks complete/manifest')
    if read(output/'complete.json').get('status')!='COMPLETE':raise RuntimeError('Original engine is incomplete')
    manifest=read(output/'manifest.json')
    def role_sha(role,required=True):
        values=[entry['sha256'] for entry in before['inputs']['files'].values() if role in entry['roles']]
        if len(values)!=1:
            if required or values:raise RuntimeError('Missing/ambiguous consumed input role: '+role)
            return None
        return values[0]
    consumed=dict(config_sha256=role_sha('wrapper/original config'),samples_sha256=role_sha('answer-free source/QA metadata'),
        model_config_sha256=role_sha('model config only; weights not rehashed'),
        references_sha256=role_sha('optional canonical-GT diagnostic references',False))
    if any(manifest.get(key)!=expected for key,expected in consumed.items()):
        raise RuntimeError('Original manifest consumed different config/data/model/reference bytes')
    required=('timing_records.jsonl','summary.json','schedule.json','length_controls.json','quiet_boundaries.jsonl') if kind=='timing' else ('records.jsonl','passes.jsonl','teacher_references.jsonl','forced_controls.json','summary.json','receipt.json')
    if any(name not in original for name in required):raise RuntimeError('Missing required original diagnostic/timer artifact')
    sidecar_name='backend_'+kind+'_sidecar.json'
    marker_name='complete_backend_timing.json' if kind=='timing' else 'complete_backend_diagnostic.json'
    sidecar=dict(status='COMPLETE',schema='trajkv.round8.backend-'+kind+'.v1',backend_module=module.__name__,backend_family=family,
        candidate_subset=True,factorial_complete=False,variant_declarations=declarations,
        class_identity=class_identity,before=before,after=after,original_artifact_sha256=original,
        consumed_input_sha256=consumed,
        wrapper_sha256=sha(wrapper_path),support_sha256=sha(__file__),
        original_artifacts_not_rewritten=True,primary_quality_eligible=False,task_accuracy_eligible=False,
        end_to_end_timing=kind=='timing',quality='not_scored',
        scope=('original quiet same-output8-token cost only; no function/allocator/CUDA-event instrumentation or accuracy claim'
            if kind=='timing' else 'original complete Full-teacher-prefix/optional separate GT diagnostics and forward accounting; no task metric or timing claim'),
        record_metadata_scope='original records are unchanged; config declarations and actual registered class are pinned here; diagnostic records need not contain per-pass refit statistics')
    write_new(output/sidecar_name,sidecar)
    if any(sha(output/name)!=expected for name,expected in original.items()):raise RuntimeError('An original engine artifact was rewritten')
    final=dict(status='COMPLETE',backend_module=module.__name__,backend_family=family,
        original_complete_sha256=original['complete.json'],original_manifest_sha256=original['manifest.json'],
        sidecar_sha256=sha(output/sidecar_name),wrapper_sha256=sha(wrapper_path),support_sha256=sha(__file__),
        config_sha256=consumed['config_sha256'],consumed_input_sha256=consumed,
        source_inputs_unchanged=True,original_artifacts_unchanged=True,primary_quality_eligible=False,task_accuracy_eligible=False,
        end_to_end_timing=kind=='timing',candidate_subset=True,factorial_complete=False,accepted_only_after_this_marker=True)
    write_new(output/marker_name,final)
    return final


def wrapper_failure(output,kind,error,before=None,after=None):
    output=Path(output)
    if not output.is_dir():return
    path=output/('backend_'+kind+'_failure.json')
    if path.exists():return
    write_new(path,dict(status='FAILED_NEW_RUN_REQUIRED',exception_type=type(error).__name__,message=str(error),
        primary_quality_eligible=False,task_accuracy_eligible=False,wrapper_complete_not_written=True,before=before,after=after))
