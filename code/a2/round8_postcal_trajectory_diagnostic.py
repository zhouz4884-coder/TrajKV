"""Bounded first-target trajectory instrument for frozen postcal, not quality.

Two old calTrain32 QAs (first Doc/first Text, first question), each identity
and frozen learned table: four trainer/deployment pairs. The trainer path
uses its original Full-teacher suffix; deployment uses question-only replay.
Only the aligned first target logits are compared. No optimizer/labels or
numerical/source changes are supported; actual CUDA is mandatory except plan.
"""
import argparse
from contextlib import contextmanager,redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT=Path(__file__).resolve().parents[2]
FREEZE_SHA='af1c0c06090522ca92efa57b0d0d85b1ae4445606510cc0bc3e1f1a11f1ff4be'
SCHEMA='trajkv.round8.postcal-first-target-trajectory-diagnostic.v1'


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def project_file(value,project_root):
    """Use actual remote paths at runtime, frozen project copies for CPU plan."""
    path=Path(value)
    if path.is_file():return path
    value=str(value).replace('\\','/')
    if '/TrajKV/' in value:return project_root/value.split('/TrajKV/',1)[1]
    return project_root/value


def frozen_files(freeze,project_root):
    result={entry['path']:sha(project_root/entry['path']) for entry in freeze['frozen_files']}
    require(len(result)==40 and all(result[x['path']]==x['sha256'] for x in freeze['frozen_files']),
        'One of the40 frozen files differs; diagnostic must not repair/tune them')
    return result


def first_target_alignment(boundary,prompt_suffix_tokens,target_tokens,suffix_tokens):
    require(all(type(x) is int for x in (boundary,prompt_suffix_tokens,target_tokens,suffix_tokens))
        and boundary>0 and prompt_suffix_tokens>0 and target_tokens>0,'Positive integer trajectory geometry required')
    require(suffix_tokens==prompt_suffix_tokens+target_tokens-1,'Teacher suffix/target alignment differs')
    first=boundary+suffix_tokens-target_tokens
    require(first==boundary+prompt_suffix_tokens-1,'Returned first target does not align to question last position')
    return dict(first_target_global_position=first,prompt_tokens=boundary+prompt_suffix_tokens,
        prefix_boundary=boundary,prompt_suffix_tokens=prompt_suffix_tokens,teacher_target_tokens=target_tokens,
        trainer_suffix_tokens=suffix_tokens,trainer_returned_logit_index=0,
        trainer_phase='question suffix plus Full teacher IDs[:-1], one batched forward with logits_to_keep=T',
        deployment_phase='question-only image-boundary replay; first margins logits before the first free token')


@contextmanager
def attribute_scope(obj,name,value):
    owned=name in vars(obj);old=vars(obj).get(name)
    setattr(obj,name,value)
    try:yield
    finally:
        if owned:setattr(obj,name,old)
        elif name in vars(obj):delattr(obj,name)


@contextmanager
def attention_scope(model):
    modules=[layer.self_attn for layer in model.model.language_model.layers]
    require(not any(hasattr(module,'_p185_read') for module in modules),'Stale attention instrument before diagnostic phase')
    try:yield
    finally:
        for module in modules:
            if hasattr(module,'_p185_read'):delattr(module,'_p185_read')


@contextmanager
def loss_first_logits(model,target_tokens,captured):
    def hook(module,args,kwargs,out):
        require(kwargs.get('logits_to_keep')==target_tokens,'Unexpected trainer logits_to_keep in instrumented forward')
        require(out.logits.ndim==3 and out.logits.shape[0]==1 and out.logits.shape[1]==target_tokens,
            'Trainer did not return batch1/T aligned logits')
        captured.append(dict(logits=out.logits[:,0].detach().float().cpu().clone(),
            output_logits_shape=list(out.logits.shape),input_ids_shape=list(kwargs['input_ids'].shape),
            position_ids_shape=list(kwargs['position_ids'].shape),logits_to_keep=target_tokens))
    handle=model.register_forward_hook(hook,with_kwargs=True)
    try:yield
    finally:handle.remove()


def cpu_fixture():
    """Exercise only the new hook/scope code with tiny CPU doubles."""
    import torch
    class Model(torch.nn.Module):
        def forward(self,input_ids,position_ids,logits_to_keep):
            return SimpleNamespace(logits=torch.arange(logits_to_keep*5).reshape(1,logits_to_keep,5).half())
    model=Model();captured=[];n=len(model._forward_hooks)
    with loss_first_logits(model,3,captured):
        model(input_ids=torch.ones(1,7,dtype=torch.long),position_ids=torch.arange(7)[None],logits_to_keep=3)
    require(len(captured)==1 and captured[0]['logits'].tolist()==[[0.,1.,2.,3.,4.]] and len(model._forward_hooks)==n,
        'CPU first-logit capture or hook restore failed')
    try:
        with loss_first_logits(model,3,[]):raise RuntimeError('intentional CPU scope exception')
    except RuntimeError:pass
    require(len(model._forward_hooks)==n,'Exception leaked forward hook')
    class Base:settings={'old':True}
    class Child(Base):pass
    old=Base.settings
    try:
        with attribute_scope(Child,'settings',{'new':True}):raise RuntimeError('intentional setting exception')
    except RuntimeError:pass
    require('settings' not in Child.__dict__ and Base.settings is old,'Exception leaked inherited class setting')
    obj=SimpleNamespace(table=lambda:'old');original=obj.table
    try:
        with attribute_scope(obj,'table',lambda:'absolute'):raise RuntimeError('intentional table exception')
    except RuntimeError:pass
    require(obj.table is original,'Exception leaked loss.table override')
    geometry=first_target_alignment(100,9,8,16)
    require(geometry['first_target_global_position']==108,'First target alignment fixture failed')
    return dict(status='PASS',CPU_only=True,first_target_slice0=True,forward_hook_restored=True,
        class_settings_and_absolute_table_restored_on_exception=True,alignment=geometry)


def prepare(freeze_path):
    freeze_path=freeze_path.resolve();require(sha(freeze_path)==FREEZE_SHA,'Exact frozen method SHA required')
    project_root=freeze_path.parents[3];freeze=read(freeze_path);before=frozen_files(freeze,project_root)
    config_path=project_root/freeze['training']['config'];cfg=read(config_path)
    train_root=(project_root/freeze['training']['complete']).parent;manifest=read(train_root/'manifest.json')
    require(sha(config_path)==manifest['config_sha256'],'Frozen actual training config differs')
    samples=project_file(cfg['arguments']['samples'],project_root)
    require(sha(samples)==manifest['samples_sha256'],'Old calTrain32 exact source bytes differ')
    grouped=read(samples)
    require(len(grouped)==32 and all(not any(key in row for key in ('answer','answers','references','ground_truth')) for row in grouped),
        'Exactly32 old answer-free train sources required')
    from round8_continuous import independent_qa
    from round8_postcal_owner import identity_table,table_values_sha256,verify_factor_artifact
    import round8_postcal_train_v2 as trainer
    import round8_postcal_runner as runner
    trainer.validate_config(cfg)
    variants={value['name']:deepcopy(value) for value in freeze['variants']}
    modes=[('identity',variants['postcal_identity_R42']),('learned',variants['postcal_learned_R42'])]
    learned=modes[1][1]
    proof=verify_factor_artifact(project_file(learned['postcal_factor_path'],project_root),
        project_file(learned['postcal_train_complete_path'],project_root),learned,
        resolver=lambda value:project_file(value,project_root))
    tables={'identity':identity_table(),'learned':proof['values']}
    sources=trainer.source_fingerprints();sources.update(runner.source_fingerprints())
    sources['round8_postcal_trajectory_diagnostic.py']=sha(__file__)
    for name,value in sources.items():
        relative='code/'+name if name.startswith('experiments/') else 'code/a2/'+name
        if relative in before:require(value==before[relative],'Actual loaded source differs from frozen method: '+relative)
    rows=[]
    for dataset in ('docvqa_grouped','textvqa'):
        original=next(row for row in grouped if row['dataset']==dataset);qa=independent_qa(original,0)
        rows.append(qa)
    pairs=[dict(pair=index,case_index=case_index,domain='doc' if row['dataset']=='docvqa_grouped' else 'text',
        mode=mode,variant=variant['name'],case_id=row['id'],source_id=row.get('source_id',row['image_id']),
        image_id=row['image_id'],independent_question_index=1,
        absolute_table_shape=[3,28,4],absolute_FP32_table_sha256=table_values_sha256(tables[mode]))
        for index,(case_index,row,mode,variant) in enumerate((i,row,mode,variant)
            for i,row in enumerate(rows) for mode,variant in modes)]
    plan=dict(status='PLAN_ONLY',schema=SCHEMA,method_freeze_sha256=FREEZE_SHA,frozen_file_count=40,
        training_config_sha256=sha(config_path),old_train_samples_sha256=sha(samples),
        old_train_manifest_sha256=sha(train_root/'manifest.json'),cases=2,pairs=pairs,pair_count=4,
        teacher_case_collections=2,collect_old_R42_calls=2,model_loads=1,source_sha256=sources,
        table_policy='Override loss.table with exact absolute FP32[3,28,4]; no log/exp or parameter fitting',
        capture_policy='Trainer model forward hook out.logits[:,0], T=case.targets.shape[1]; deployment first MarginsHook logits SHA must equal result.initial_logits_sha256',
        deployment_arguments=dict(turns=1,max_tokens=1,max_patches=8,compress_before_question=True,
            margin=0.,max_replays=0,save_states=False,read_budget_bytes=None,gain_table_data=None),
        limits='4 old-train diagnostic pairs; no numerical agreement threshold, no tuning/checkpoint/quality metric, no confirmation data',
        phase_caveat='Causal first target aligns ideally, but full teacher suffix and question-only batched FP16 GEMM shapes can differ numerically',
        GPU_executed=False,model_forward_executed=False,GT_references_opened=False,confirmation_samples_opened=False)
    return plan,freeze,cfg,rows,modes,tables,project_root,before,manifest


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_frozen_signature(model):
    states=[dict(name=name,version=value._version,shape=list(value.shape),dtype=str(value.dtype),requires_grad=value.requires_grad,
        gradient_absent=value.grad is None) for name,value in model.named_parameters()]
    require(all(not x['requires_grad'] and x['gradient_absent'] for x in states),'Model weights/gradients are not frozen')
    return dict(parameter_count=len(states),version_state_sha256=object_sha(states),
        scope='Frozen requires_grad/gradient and in-place version-counter identity; not a hash of all weight bytes')


def runtime_inputs(cfg,rows,project_root,training_manifest):
    args=cfg['arguments'];paths={str(project_file(args['samples'],project_root)),str(project_file(Path(args['model'])/'config.json',project_root))}
    for row in rows:
        for path,expected in zip(row['remote_images'],row['hashes']):
            require(sha(path)==expected,'Old train image bytes differ');paths.add(path)
    paths.update(str(path) for path in Path(args['legacy_scripts']).glob('*.py'))
    paths.add(args['kivi_source'])
    result={path:sha(path) for path in sorted(paths)}
    frozen=training_manifest['input_sha256']
    for path,value in result.items():
        require(path in frozen and frozen[path]==value,'Runtime old-training external bytes changed: '+path)
    return result


def _execute(plan,freeze,cfg,rows,modes,tables,project_root,before,training_manifest,output):
    import torch
    require(torch.cuda.is_available(),'Actual CUDA mandatory; use --plan-only for CPU')
    sys.path.insert(0,str(ROOT/'code/experiments'))
    from bank import install_legacy
    install_legacy(cfg['arguments']['legacy_scripts'])
    from p180_multiturn_core import attention_bridge
    import run_p175_modern_replication as oldbase
    from round6_student_codec import teacher_case
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from round8_continuous import MarginsHook
    import round8_postcal_train_v2 as trainer
    import round8_postcal_runner as runner
    import session_boundary
    torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    input_before=runtime_inputs(cfg,rows,project_root,training_manifest)
    args=cfg['arguments'];processor=AutoProcessor.from_pretrained(args['model'],local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(args['model'],local_files_only=True,dtype=torch.float16,
        device_map='cuda:0',attn_implementation='sdpa').eval()
    for value in model.parameters():value.requires_grad_(False)
    model_before=model_frozen_signature(model);model_config=model.config.get_text_config()
    require(model_config.num_hidden_layers==28 and model_config.num_key_value_heads==4,'Expected frozen28x4 geometry')
    book=torch.tensor(read(cfg['bank_settings']['direction_book_path'])['centers'],device=model.device,dtype=torch.float16)
    parameters=[torch.nn.Parameter(torch.zeros((28,4),device=model.device,dtype=torch.float32),requires_grad=False) for _ in range(3)]
    loss=trainer.PackedKLLoss(model,model_config,book,parameters,cfg);cases=[];contexts=[]
    with torch.no_grad():
        for row in rows:
            with attention_scope(model):
                case=teacher_case(model,processor,row,args,8,attention_bridge,oldbase.stop_ids(model,processor))
                case['legacy_scripts'],case['kivi_source']=args['legacy_scripts'],args['kivi_source']
                encoded,budget=trainer.collect_old_R42(case,model,model_config,book,cfg['bank_settings'])
                case['encoded'],case['budget']=encoded,budget;case=trainer.move(case,'cpu')
            alignment=first_target_alignment(int(case['boundary']),int(case['prompt_suffix_tokens']),int(case['targets'].shape[1]),int(case['suffix'].shape[1]))
            visual=case['visual'].tolist()
            contexts.append(dict(case_id=row['id'],image_id=row['image_id'],alignment=alignment,real_Nv=len(visual),
                visual_indices=visual,visual_indices_sha256=tensor_sha(case['visual']),
                teacher_target_ids_sha256=tensor_sha(case['targets']),teacher_logits_sha256=tensor_sha(case['teacher_logits']),
                trainer_suffix_ids_sha256=tensor_sha(case['suffix']),prefix_layers=len(case['prefix']),
                prefix_shapes=[dict(key=list(k.shape),value=list(v.shape)) for k,v in case['prefix']],
                encoded_layers=[trainer.state_fingerprint(SimpleNamespace(**entry)) for entry in encoded],budget=budget))
            cases.append(case)
        results=[]
        for pair in plan['pairs']:
            row=rows[pair['case_index']];case=cases[pair['case_index']];mode=pair['mode'];variant=next(v for m,v in modes if m==mode)
            table=torch.tensor(tables[mode],device=model.device,dtype=torch.float32).contiguous()
            require(tensor_sha(table)==pair['absolute_FP32_table_sha256'],'Absolute FP32 table changed in tensor construction')
            captured=[]
            with attention_scope(model),attribute_scope(loss,'table',lambda table=table:table),loss_first_logits(model,int(case['targets'].shape[1]),captured):
                diagnostic_KL=float(loss(case,identity=mode=='identity'))
            require(len(captured)==1,'Expected exactly one instrumented trainer model forward')
            trainer_logits=captured[0].pop('logits');deployment_logits=[];feature_rows=[]
            original_features=session_boundary.features
            def features(processor,source,max_patches):
                require(source['id']==row['id'] and max_patches==8,'Deployment QA/cap differs')
                value=original_features(processor,source,max_patches);indices=(value['input_ids'][0]==model.config.image_token_id).nonzero().flatten()
                feature_rows.append(dict(prompt_tokens=int(value['input_ids'].shape[-1]),prefix_boundary=int(indices[-1])+1,
                    real_Nv=int(indices.numel()),visual_indices=indices.tolist(),visual_indices_sha256=tensor_sha(indices),
                    input_ids_sha256=tensor_sha(value['input_ids']),input_ids_shape=list(value['input_ids'].shape)))
                return value
            def first_margin(logits):
                if not deployment_logits:deployment_logits.append(logits.detach().float().cpu().clone())
            runargs=SimpleNamespace(**args)
            for key,value in plan['deployment_arguments'].items():setattr(runargs,key,value)
            for key in ('model','samples','legacy_scripts','kivi_source'):setattr(runargs,key,Path(getattr(runargs,key)))
            # Registration and settings are scoped independently; session
            # stdout includes train answers, so do not persist that stream.
            with attention_scope(model),runner.backend_registration('postcal') as cls:
                with attribute_scope(cls,'settings',variant),attribute_scope(session_boundary,'features',features),MarginsHook(session_boundary,first_margin),redirect_stdout(io.StringIO()):
                    produced,snapshots=session_boundary.run_session(model,processor,row,runargs,'round8_joint','base',keep_reference=False)
            require(len(deployment_logits)==len(feature_rows)==1 and not snapshots,'Deployment capture or no-snapshot contract differs')
            actual=deployment_logits[0]
            require(tensor_sha(actual)==produced['initial_logits_sha256'],'Deployment margins capture is not initial_logits SHA')
            require(produced['postcal_table_values_sha256']==pair['absolute_FP32_table_sha256'],'Deployment table differs from direct trainer absolute table')
            geometry=contexts[pair['case_index']];feature=feature_rows[0]
            require(feature['prompt_tokens']==geometry['alignment']['prompt_tokens'] and feature['prefix_boundary']==case['boundary']
                and feature['visual_indices']==geometry['visual_indices'] and produced['visual_tokens']==geometry['real_Nv'],
                'Trainer/deployment source prompt/prefix/visual geometry differs')
            require(tuple(trainer_logits.shape)==tuple(actual.shape) and bool(torch.isfinite(trainer_logits).all()) and bool(torch.isfinite(actual).all()),
                'First-target logit shape/finite contract differs')
            difference=trainer_logits-actual
            result=dict(pair,trainer_capture=captured[0],deployment_features=feature,logit_shape=list(actual.shape),
                trainer_first_logits_FP32_sha256=tensor_sha(trainer_logits),deployment_first_logits_FP32_sha256=tensor_sha(actual),
                deployment_initial_logits_sha256=produced['initial_logits_sha256'],deployment_capture_SHA_verified=True,
                table_tensor_FP32_sha256=tensor_sha(table),trainer_diagnostic_teacher_KL=diagnostic_KL,
                max_abs=float(difference.abs().max()),relative_L2=float(difference.norm()/actual.norm().clamp_min(1e-20)),
                trainer_top1_id=int(trainer_logits.argmax(-1).item()),deployment_top1_id=int(actual.argmax(-1).item()),
                top1_equal=bool(torch.equal(trainer_logits.argmax(-1),actual.argmax(-1))),bitwise_equal=bool(torch.equal(trainer_logits,actual)),
                compared_phase=geometry['alignment'],agreement_threshold_enforced=False,task_accuracy_eligible=False,
                deployment_owned_initial=produced['owned_initial'],source_and_absolute_table_identity_verified=True)
            results.append(result);print(json.dumps(dict(pair=pair['pair'],domain=pair['domain'],mode=mode,max_abs=result['max_abs'],relative_L2=result['relative_L2'],top1_equal=result['top1_equal'])),flush=True)
            require(model_frozen_signature(model)==model_before,'Model parameter version/gradient state changed')
            require(not any(hasattr(module,'_p185_read') for module in loss.modules),'Attention hook leaked between diagnostic pairs')
    require(frozen_files(freeze,project_root)==before,'Frozen40 files changed during diagnostic')
    require(runtime_inputs(cfg,rows,project_root,training_manifest)==input_before,'Runtime input/source image bytes changed')
    require(trainer.source_fingerprints() | runner.source_fingerprints() | {'round8_postcal_trajectory_diagnostic.py':sha(__file__)}==plan['source_sha256'],
        'Actual diagnostic source bytes changed')
    report=dict(status='COMPLETE_DIAGNOSTIC_ONLY',schema=SCHEMA,pair_count=4,case_count=2,contexts=contexts,pairs=results,
        method_freeze_sha256=FREEZE_SHA,frozen_files_before=before,frozen_files_after=frozen_files(freeze,project_root),
        source_sha256=plan['source_sha256'],input_sha256_before=input_before,input_sha256_after=input_before,
        frozen_model_before=model_before,frozen_model_after=model_frozen_signature(model),model_loads=1,
        teacher_case_collections=2,old_R42_collections=2,absolute_FP32_table_no_log_exp_roundtrip=True,
        instrumentation_restored=False,GPU_executed=True,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        task_accuracy_eligible=False,ANLS_or_TextVQA_scored=False,GT_references_opened=False,confirmation_samples_opened=False,
        agreement_scope='First-target phase comparison only; full teacher-forced suffix and question-only deployment may differ under batched FP16 arithmetic. No bitwise/quality guarantee or tuning follows.')
    return report


def execute(plan,freeze,cfg,rows,modes,tables,project_root,before,training_manifest,output):
    """Restore registration/settings even if a frozen helper exits early."""
    import torch
    require(torch.cuda.is_available(),'Actual CUDA mandatory; use --plan-only for CPU')
    sys.path.insert(0,str(ROOT/'code/experiments'))
    from bank import install_legacy
    install_legacy(cfg['arguments']['legacy_scripts'])
    # Complete the existing registration chain before recording its baseline.
    import round8_runner  # noqa: F401
    import round8_bank
    import factory
    import run_representation_variants as executor
    import session
    import session_boundary
    pointers=(factory.bank_class,executor.bank_class,session.ProgressiveBank,session_boundary.ProgressiveBank,
        session_boundary.features,session_boundary.margins)
    cls=round8_bank.Round8Bank;owned='settings' in cls.__dict__;settings=getattr(cls,'settings')
    try:report=_execute(plan,freeze,cfg,rows,modes,tables,project_root,before,training_manifest,output)
    finally:
        factory.bank_class,executor.bank_class,session.ProgressiveBank,session_boundary.ProgressiveBank,session_boundary.features,session_boundary.margins=pointers
        if owned:cls.settings=settings
        elif 'settings' in cls.__dict__:delattr(cls,'settings')
    require((factory.bank_class,executor.bank_class,session.ProgressiveBank,session_boundary.ProgressiveBank,
        session_boundary.features,session_boundary.margins)==pointers,'Registration/instrumentation pointers did not restore')
    require(('settings' in cls.__dict__)==owned and getattr(cls,'settings') is settings,'Round8Bank inherited settings did not restore')
    report['instrumentation_restored']=True
    report['restored_state_scope']='forward hooks, margins, features, module attention readers, absolute loss.table, factory/executor/session registration, inherited Round8Bank settings'
    write_new(output/'diagnostic.json',report)
    write_new(output/'complete.json',dict(status='COMPLETE_DIAGNOSTIC_ONLY',pairs=4,GPU_executed=True,task_accuracy_eligible=False,
        diagnostic_sha256=sha(output/'diagnostic.json'),plan_sha256=sha(output/'plan.json'),frozen40_unchanged=True,
        instrumentation_restored=True))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,default=ROOT/'configs/a2/round8_v1/method_freeze_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    plan,freeze,cfg,rows,modes,tables,project_root,before,manifest=prepare(a.freeze)
    plan['CPU_instrument_fixture']=cpu_fixture()
    require(frozen_files(freeze,project_root)==before,'Frozen files changed during local plan')
    a.output.mkdir(parents=True,exist_ok=False);write_new(a.output/'plan.json',plan)
    if a.plan_only:
        write_new(a.output/'complete_plan.json',dict(status='PLAN_ONLY',pair_count=4,plan_sha256=sha(a.output/'plan.json'),GPU_executed=False))
        print(json.dumps(dict(status='PLAN_ONLY',pair_count=4,cases=2,GPU_executed=False,model_forward_executed=False,source_sha256=sha(__file__),
            CPU_instrument_fixture='PASS',frozen40_SHA_unchanged=True,output=str(a.output))),flush=True);return
    try:execute(plan,freeze,cfg,rows,modes,tables,project_root,before,manifest,a.output)
    except BaseException as error:
        write_new(a.output/'failure.json',dict(status='FAILED_DIAGNOSTIC_NEW_OUTPUT_REQUIRED',exception_type=type(error).__name__,message=str(error),
            frozen40_unchanged=frozen_files(freeze,project_root)==before,task_accuracy_eligible=False,GT_references_opened=False))
        raise


if __name__=='__main__':main()
