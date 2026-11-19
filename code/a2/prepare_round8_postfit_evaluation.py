"""Prepare fixed endpoint-book R8 selection QA configs/jobs, without launch.

Plan-only works before fitting. Generation requires both complete_books.json
receipts and verifies every listed book's exact JSON SHA, FP16 payload SHA,
geometry and source/mask provenance. No fit score chooses an endpoint.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from round8_data_prepare import ROOT,read,sha,new
from round8_budget_check import cost
from round8_quality_owner import local_path

FIT_PLAN = ROOT/'configs/a2/round8_v1/native_fit_plan_v1.json'
RUNTIME_JOB = ROOT/'configs/jobs/r8_native_collect_cal32_select32_gpu0_diskverify_v2.json'
SHAPES = {'k_global4_16':(16,4),'k_global8_256':(256,8),
          'k_position4_16':(32,16,4),'v_global8_256':(256,8)}


def require(value,message):
    if not value: raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def load_context(a):
    fit_plan=read(FIT_PLAN)
    smoke_path=ROOT/'configs/a2/round8_v1/identity_smoke6.json'
    smoke=read(smoke_path)
    baseline_path=ROOT/'configs/a2/round7_v1/confirm_doc128_s0.json'
    baseline_config=read(baseline_path)
    baseline=next(v for v in baseline_config['variants'] if v['name']=='r7_joint_k1_gt_cond')
    require(len(baseline['selected_stages'])==224 and baseline['res_stages_limit']==43,'Unexpected frozen priority contract')
    template=next(v for v in smoke['variants'] if v['name']=='r8_oldK_binaryV_identity')
    require(template['selected_stages']==baseline['selected_stages'] and template['res_stages_limit']==43,
        'Smoke/frozen priority mismatch')
    global_root=(a.remote_root or fit_plan['provenance']['source_roots']['model_data_results']).rstrip('/')
    source_root=(a.source_root or fit_plan['provenance']['source_roots']['isolated_code']).rstrip('/')
    require(global_root.startswith('/') and source_root.startswith('/') and source_root!=global_root,
        'Independent absolute POSIX source root and project data/results root required')
    require(not any(root=='/tmp' or root.startswith('/tmp/') or root.startswith('/home/') for root in (global_root,source_root)),
        'Remote new source/results roots must stay on the project data disk')
    split_path=ROOT/'configs/a2/round8_v1/calibration_source_splits_v1.json'
    split=read(split_path)
    require(len(split['train'])==len(split['select'])==32 and not set(split['train'])&set(split['select']),
        'Expected disjoint main cal32/select32 source contract')
    views=read(ROOT/'audit/a2/round8_exposed_evaluation_views_v1.json')['files']
    for kind,questions in [('doc',38),('text',22)]:
        spec=views[kind]; samples=ROOT/spec['samples']; rows=read(samples)
        require(sha(samples)==spec['samples_sha256'] and sha(ROOT/spec['references'])==spec['references_sha256'],
            'Exposed evaluation input SHA changed')
        require(len(rows)==questions and len({r['image_id'] for r in rows})==16 and all(r['turns']==1 for r in rows),
            'Use independent QA selection views, not grouped Text calibration sessions')
        source_ids={r['source_id'] for r in rows}
        require(source_ids<=set(split['select']) and not source_ids&set(split['train']),'Evaluation source leakage')
    require(sha(local_path(baseline['direction_book_path']))==fit_plan['provenance']['old_book_sha256'],
        'Frozen old book differs from fit plan')
    require(sha(local_path(baseline['metadata_factor_path']))==fit_plan['provenance']['frozen_factors_sha256'],
        'Frozen factor differs from fit plan')
    runtime_job=read(RUNTIME_JOB)
    worker_python=runtime_job['argv'][0]
    require(worker_python.startswith(global_root.rsplit('/',1)[0]+'/_runtime/')
        and worker_python.endswith('/envs/p72_modern_vlm/bin/python') and not worker_python.startswith('/home/'),
        'Worker must use the validated data-volume P72 runtime, not the old home prefix')
    require(runtime_job.get('round8_source_root')==source_root,'Runtime reference job/source-root mismatch')
    return dict(fit_plan=fit_plan,smoke=smoke,smoke_path=smoke_path,baseline=baseline,
        baseline_path=baseline_path,baseline_config=baseline_config,template=template,global_root=global_root,
        source_root=source_root,split_path=split_path,split=split,views=views,
        runtime_job=runtime_job,worker_python=worker_python)


def expected_paths(a,c):
    dirs={'k':a.k_fit_dir or local_path(c['fit_plan']['jobs']['K']['output']),
          'v':a.v_fit_dir or local_path(c['fit_plan']['jobs']['V']['output'])}
    expected={}
    for axis in ('k','v'):
        assets=c['fit_plan']['jobs'][axis.upper()]['assets']
        expected[axis]=dict(local=str(dirs[axis]),remote=c['global_root']+'/results/round8_v1/'+
            ('native_books_k_v1' if axis=='k' else 'native_books_v_v1'),
            complete=str(dirs[axis]/'complete_books.json'),files=[asset+'_'+mode.replace('-','_')+'.json'
                for asset in assets for mode in c['fit_plan']['modes']])
    return expected


def verified_books(a,c,expected):
    found={}; receipts={}
    dump_path=a.dump_manifest or local_path(c['fit_plan']['dump'])/'manifest.json'
    require(dump_path.is_file(),'Native dump manifest required to verify frozen factors and actual mask: '+str(dump_path))
    dump=read(dump_path)
    require(dump['schema']=='trajkv.round8.native-dump.v1' and dump['labels_used'] is False
        and dump['model_weights_frozen'] is True and dump['diagnostic_source_owner'] is True and dump['deployable'] is False
        and dump['frozen_factors_sha256']==sha(local_path(c['baseline']['metadata_factor_path']))
        and dump['old_book_sha256']==sha(local_path(c['baseline']['direction_book_path']))
        and dump['selected_stages']==c['baseline']['selected_stages'][:43],
        'Native dump factor/old-book/mask contract mismatch')
    for axis,spec in expected.items():
        folder=Path(spec['local']); complete=folder/'complete_books.json'
        require(complete.is_file(),'Fit not complete; use --plan-only until available: '+str(complete))
        receipt=read(complete); provenance=receipt['provenance']
        require(receipt['status']=='COMPLETE' and receipt['schema']=='trajkv.round8.source-balanced-native-fit.v1',
            'Invalid final-book completion receipt')
        require(provenance['source_split_sha256']==sha(c['split_path'])
            and provenance['old_encoder_book_sha256']==sha(local_path(c['baseline']['direction_book_path']))
            and provenance['training_source_ids']==c['split']['train'] and provenance['select_source_ids']==c['split']['select']
            and provenance['fixed_stage_mask']==c['baseline']['selected_stages'][:43]
            and provenance['dump_manifest_sha256']==sha(dump_path)
            and provenance['labels_used'] is False and provenance['model_weights_frozen'] is True,
            'Fit train/select/old-book/stage/label provenance mismatch')
        require(receipt.get('numerical_check',{}).get('status')=='PASS','Final fitter CPU algebra check missing/failed')
        indexed={}
        for entry in receipt['books']:
            key=(entry['asset'],entry['mode']); filename=entry['file']
            require(key not in indexed and Path(filename).name==filename and entry['asset'].startswith(axis+'_'),
                'Duplicate, foreign-axis or nonlocal final book entry')
            path=folder/filename; require(path.is_file() and sha(path)==entry['sha256'],'Final endpoint exact JSON SHA mismatch: '+str(path))
            book=read(path); centers=np.asarray(book['centers'],dtype=np.float16)
            payload=centers.astype('<f2',copy=False).tobytes(order='C')
            payload_sha=hashlib.sha256(payload).hexdigest()
            require(tuple(centers.shape)==SHAPES[entry['asset']] and np.isfinite(centers).all()
                and entry['shape']==list(centers.shape) and book['storage_dtype']=='float16',
                'Final book geometry/storage mismatch')
            require(entry['actual_book_bytes']==book['actual_book_bytes']==centers.nbytes
                and entry['centers_fp16_le_sha256']==book['centers_fp16_le_sha256']==payload_sha,
                'Final book physical FP16 payload fingerprint mismatch')
            require(book['asset']==entry['asset'] and book['fit_mode']==entry['mode'] and book['axis']==axis,
                'Final book asset/mode mismatch')
            require(all(book.get(k)==value for k,value in provenance.items()),'Final book and completion provenance differ')
            indexed[key]=dict(local=str(path),remote=spec['remote']+'/'+filename,sha256=entry['sha256'],
                centers_fp16_le_sha256=payload_sha,shape=list(centers.shape),actual_book_bytes=centers.nbytes,
                asset=entry['asset'],fit_mode=entry['mode'])
        required={(asset,mode) for asset in c['fit_plan']['jobs'][axis.upper()]['assets'] for mode in c['fit_plan']['modes']}
        require(set(indexed)==required,'Final completion receipt is partial or includes unexpected endpoint identities')
        found.update(indexed)
        receipts[axis]=dict(file=str(complete),sha256=sha(complete),remote=spec['remote']+'/complete_books.json',
            provenance=provenance,books=list(indexed.values()),dump_manifest=str(dump_path),dump_manifest_sha256=sha(dump_path))
    require(receipts['k']['provenance']==receipts['v']['provenance'],'K/V fits did not consume the identical frozen source/mask contract')
    return found,receipts


def make_variants(a,c,assets):
    base=c['baseline']; old=copy.deepcopy(base); old['name']='r7_frozen_joint43_control'
    full=copy.deepcopy(next(v for v in c['baseline_config']['variants'] if v['method']=='full'))
    template=copy.deepcopy(c['template'])
    k8=assets[('k_global8_256','reencode')]['remote']; v8=assets[('v_global8_256','reencode')]['remote']
    notes={}
    def variant(name,book=None,value=None,key_mode='reencode',value_mode='binary',joint=False):
        v=copy.deepcopy(template); v['name']=name
        for key in ('encoder_book_path','value_joint_book_path','joint_gain_refit'):
            v.pop(key,None)
        if book: v['direction_book_path']=book
        v.update(key_encoding=key_mode,value_encoding=value_mode)
        if value: v['value_joint_book_path']=value
        if key_mode=='decoder-only': v['encoder_book_path']=base['direction_book_path']
        if joint: v['joint_gain_refit']=True
        require(v['selected_stages']==base['selected_stages'] and v['res_stages_limit']==43,'Priority changed in variant generation')
        return v
    variants=[full,old,
        variant('r8_k4old_vbinary'),
        variant('r8_k8re_vbinary',book=k8),
        variant('r8_k4old_v8re',value=v8,value_mode='reencode'),
        variant('r8_k8re_v8re',book=k8,value=v8,value_mode='reencode'),
        variant('r8_k4fit_vbinary',book=assets[('k_global4_16','reencode')]['remote']),
        variant('r8_k4pos_vbinary',book=assets[('k_position4_16','reencode')]['remote']),
        variant('r8_k4old_vbinary_jointgain',joint=True)]
    mode='reencode' if a.decoder_book_policy=='matched-reencode' else 'decoder-only'
    if 'k' in a.decoder_controls:
        variants.append(variant('r8_k8dec_vbinary',book=assets[('k_global8_256',mode)]['remote'],key_mode='decoder-only'))
    if 'v' in a.decoder_controls:
        variants.append(variant('r8_k4old_v8dec',value=assets[('v_global8_256',mode)]['remote'],value_mode='decoder-only'))
    require(len(variants)==9+len(a.decoder_controls) and len({v['name'] for v in variants})==len(variants),'Unexpected arm coverage')
    by={v['name']:v for v in variants}
    require(by['r8_k8re_vbinary']['direction_book_path']==by['r8_k8re_v8re']['direction_book_path']
        and by['r8_k4old_v8re']['value_joint_book_path']==by['r8_k8re_v8re']['value_joint_book_path'],'Main four-cell endpoints not shared')
    notes.update(main_factorial='Same learned K8 reencode endpoint shared across both K8 cells; same learned V8 reencode endpoint shared across both V8 cells; residuals regenerated from each candidate base.',
        decoder_controls=('Same reencode-fitted book, different deployment field policy; isolates deployment encoding/residual policy at fixed centers. Training metadata describes fit, not deployment ownership.'
            if mode=='reencode' else 'Separately fitted decoder-only endpoint and field policy both differ; a joint fit/encoding comparison, not pure encoding attribution.'),
        joint_gain='Old K/V fields with existing selected K gain+alpha refit, no additional storage fields.',
        ranking='Full frozen 224-entry ranking retained (448 B); actual fixed 43-stage prefix (23K,20V).')
    return variants,notes


def budget_rows(variants,c,assets):
    lookup={v['remote']:v for v in assets.values()}; rows=[]
    for nv in (768,1024,1280,1792):
        for variant in variants:
            if variant['method']=='full':
                denominator=28*4*nv*128*4
                position=nv*8
                rows.append(dict(variant=variant['name'],visual_tokens=nv,role='FP16 reference',
                    visual_FP16_denominator_bytes=denominator,visual_base_bytes=denominator,
                    shared_visual_positions=position,complete_visual_owner_bytes=denominator+position,
                    nominal_payload_bits_per_KV_scalar=16,
                    complete_bits_per_KV_scalar=16*(denominator+position)/denominator,
                    scope='L28/Hkv4/D128 full K+V plus shared int64 visual positions; actual runtime owner is audited independently from records'))
                continue
            shape=lookup.get(variant['direction_book_path'],{}).get('shape',[16,4])
            positions=shape[0] if len(shape)==3 else 1
            value8=variant.get('value_encoding','binary')!='binary'
            row=cost(nv,28,4,shape[-1],shape[-2],8 if value8 else None,256 if value8 else None,
                kpos=positions,decoder_only=variant.get('key_encoding')=='decoder-only',stages=43,ranking_entries=224)
            row['variant']=variant['name']; rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--plan-output',type=Path)
    p.add_argument('--k-fit-dir',type=Path); p.add_argument('--v-fit-dir',type=Path)
    p.add_argument('--dump-manifest',type=Path,help='Downloaded native manifest only; native .pt tensors are not opened')
    p.add_argument('--source-root'); p.add_argument('--remote-root')
    p.add_argument('--name','--output-prefix',dest='output_prefix',default='postfit_select11_v2',
        help='New versioned config/job/output prefix; existing v1 evidence is never overwritten')
    p.add_argument('--receipt',type=Path)
    p.add_argument('--decoder-controls',nargs='+',choices=('k','v'),default=['k','v'])
    p.add_argument('--decoder-book-policy',choices=('matched-reencode','fitted-decoder-only'),default='matched-reencode')
    a=p.parse_args(); require(re.fullmatch(r'[A-Za-z0-9_]+',a.output_prefix),'Simple new output prefix required')
    require(len(set(a.decoder_controls))==len(a.decoder_controls),'Duplicate decoder control')
    c=load_context(a); expected=expected_paths(a,c); count=9+len(a.decoder_controls)
    plan=dict(schema='trajkv.round8.postfit-selection-plan.v1',status='PLAN_ONLY_NOT_EXECUTED',
        expected_fits=expected,missing_files=[str(Path(v['local'])/name) for v in expected.values()
            for name in ['complete_books.json']+v['files'] if not (Path(v['local'])/name).is_file()],
        source_root=c['source_root'],model_data_results_root=c['global_root'],output_prefix=a.output_prefix,
        path_contract=dict(code_root=c['source_root'],config_sample_asset_result_root=c['global_root'],
            worker_python=c['worker_python'],runtime_reference_job=str(RUNTIME_JOB.relative_to(ROOT)),
            runtime_reference_job_sha256=sha(RUNTIME_JOB),round8_source_root_required=True,
            PYTHONDONTWRITEBYTECODE='1',new_caches_under=c['global_root']+'/cache/round8_v1/'),
        required_native_manifest=str(a.dump_manifest or local_path(c['fit_plan']['dump'])/'manifest.json'),
        variants=count,doc_questions=38,text_questions=22,doc_sources=16,text_sources=16,
        expected_records_doc=38*count,expected_records_text=22*count,expected_answers_total=60*count,
        decoder_controls=a.decoder_controls,decoder_book_policy=a.decoder_book_policy,
        expected_full_ranking=224,actual_residual_stages=43,ranking_bytes=448,
        no_evaluation_files_generated=a.plan_only,no_fitted_book_selected_by_score=True,
        exposure_scope='Previously exposed selection sources; developmental QA, not blind confirmation',
        future_confirmation_answers_opened=False,GPU_used=False,uploaded=False,code_sha256=sha(__file__))
    if a.plan_only:
        if a.plan_output: new(a.plan_output,plan)
        print(json.dumps(dict(status=plan['status'],variants=count,questions_total=60,expected_answers=60*count,
            missing_fit_files=len(plan['missing_files']),plan_output=str(a.plan_output) if a.plan_output else None,GPU_used=False)))
        return
    assets,fit_receipts=verified_books(a,c,expected)
    variants,notes=make_variants(a,c,assets)
    outputs=[]; targets=[]
    receipt_path=a.receipt or ROOT/f'audit/a2/round8_{a.output_prefix}_prepared.json'
    for kind in ('doc','text'):
        for prefix in ('','job_'):
            targets.append(ROOT/f'configs/a2/round8_v1/{prefix}{a.output_prefix}_{kind}.json')
    require(not any(path.exists() for path in targets+[receipt_path]),'Output already exists; preserve it and choose a new prefix/receipt')
    old_job=c['runtime_job']
    for kind,gpu in [('doc',0),('text',1)]:
        spec=c['views'][kind]
        config=copy.deepcopy(c['smoke']); config.update(start=0,count=spec['questions'],variants=variants)
        config['arguments'].update(samples=c['global_root']+'/'+spec['samples'].replace('\\','/'),turns=1)
        config['scope']='R8 historical selection independent QA only; fixed final endpoints; no confirmation or blind claim'
        config['postfit_provenance']=dict(fit_complete_sha256={axis:row['sha256'] for axis,row in fit_receipts.items()},
            used_books={row['remote']:row['sha256'] for row in assets.values()
                if any(row['remote']==variant.get(key) for variant in variants for key in ('direction_book_path','value_joint_book_path'))},
            decoder_book_policy=a.decoder_book_policy,
            source_split_sha256=sha(c['split_path']),selection_scores_used_to_choose_endpoint=False,
            sample_sha256=spec['samples_sha256'],analysis_references_local_only=spec['references'])
        cp=ROOT/f'configs/a2/round8_v1/{a.output_prefix}_{kind}.json'; new(cp,config)
        job=dict(schema='trajkv.round8.job.v2',name='round8_'+a.output_prefix+'_'+kind,
            cwd=c['source_root'],round8_source_root=c['source_root'],env=copy.deepcopy(old_job['env']))
        job['env']['CUDA_VISIBLE_DEVICES']=str(gpu)
        job['env']['PYTHONDONTWRITEBYTECODE']='1'
        cache=c['global_root']+'/cache/round8_v1/'+job['name']
        for key,leaf in [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),
                ('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('PYTHONPYCACHEPREFIX','pycache'),
                ('TORCH_HOME','torch'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('PIP_CACHE_DIR','pip'),
                ('UV_CACHE_DIR','uv'),('XDG_CACHE_HOME','xdg')]:
            job['env'][key]=cache+'/'+leaf
        hf=c['global_root']+'/cache/round8_v1/huggingface'
        job['env'].update(HF_HOME=hf,HF_HUB_CACHE=hf+'/hub',HUGGINGFACE_HUB_CACHE=hf+'/hub',HF_MODULES_CACHE=hf+'/modules')
        output=c['global_root']+'/results/round8_v1/'+a.output_prefix+'_'+kind
        job['argv']=[c['worker_python'],c['source_root']+'/code/a2/round8_runner.py','--config',
            c['global_root']+'/'+cp.relative_to(ROOT).as_posix(),'--output',output]
        job['runtime_policy']='Validated data-volume P72 worker; sourceRoot is code only, configs/samples/assets/results stay under explicit globalR'
        job['prerequisites']=['Root verifies final source snapshot, uploaded config/sample/asset SHA and idle GPU before launch',
            'Both final-book completion receipts and all endpoint hashes were verified locally by this preparer']
        jp=ROOT/f'configs/a2/round8_v1/job_{a.output_prefix}_{kind}.json'; new(jp,job)
        outputs.append(dict(dataset=kind,gpu=gpu,config=str(cp.relative_to(ROOT)),config_sha256=sha(cp),
            job=str(jp.relative_to(ROOT)),job_sha256=sha(jp),output=output,samples=spec['samples'],
            samples_sha256=spec['samples_sha256'],references_local_only=spec['references'],
            references_sha256=spec['references_sha256'],sources=16,independent_questions=spec['questions'],
            expected_records=spec['questions']*count,expected_answers=spec['questions']*count))
    receipt=dict(plan,status='PREPARED_NOT_EXECUTED',created_at_utc=datetime.now(timezone.utc).isoformat(),
        no_evaluation_files_generated=False,outputs=outputs,fit_receipts=fit_receipts,book_artifacts=list(assets.values()),
        input_fit_plan_sha256=sha(FIT_PLAN),input_smoke_config_sha256=sha(c['smoke_path']),
        frozen_baseline_config_sha256=sha(c['baseline_path']),source_split_sha256=sha(c['split_path']),
        requested_ranking_sha256=canonical_hash(c['baseline']['selected_stages']),
        actual_mask_sha256=canonical_hash(c['baseline']['selected_stages'][:43]),
        method_notes=notes,budget_contracts=budget_rows(variants,c,assets),
        execution_scope='Configs/jobs only. No GPU, SSH, upload, task scores or timing measurements produced.')
    new(receipt_path,receipt)
    print(json.dumps(dict(status=receipt['status'],variants=count,independent_questions=60,expected_answers=60*count,
        configs=[row['config'] for row in outputs],jobs=[row['job'] for row in outputs],receipt=str(receipt_path),GPU_used=False)))


if __name__=='__main__': main()
