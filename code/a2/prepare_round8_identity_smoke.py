"""Prepare six-arm R8 identity smoke; no training, SSH or GPU execution."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from round8_data_prepare import ROOT,read,sha,new
from round8_budget_check import cost
from round8_quality_owner import local_path


def main():
    original_path=ROOT/'configs/a2/round7_v1/confirm_doc128_s0.json'
    original=read(original_path)
    base=copy.deepcopy(next(v for v in original['variants'] if v['name']=='r7_joint_k1_gt_cond'))
    assert len(base['selected_stages'])==224 and base['res_stages_limit']==43
    global_root=original['arguments']['samples'].split('/configs/',1)[0]
    stage_root=global_root+'/stages/round8_20261118_v1'
    asset_dir=ROOT/'results/round8_v1/initial_assets'
    asset_dir.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'configs/a2/round8_v1/identity_smoke6.json').exists()
    old_path=local_path(base['direction_book_path']); old=read(old_path)
    centers=np.asarray(old['centers'],dtype=np.float16)
    assert centers.shape==(16,4)
    old_copy=asset_dir/'old_gaussian_g4_16_exact.json'
    with old_copy.open('xb') as stream: stream.write(old_path.read_bytes())
    assert sha(old_copy)==sha(old_path)
    # Low nibble encodes the first4 coordinates, high nibble the next4.
    product=np.asarray([np.concatenate((centers[j&15],centers[j>>4])) for j in range(256)],dtype=np.float16)
    cube=np.asarray([[(j>>bit)&1 for bit in range(8)] for j in range(256)],dtype=np.float16)
    k8=asset_dir/'cartesian_g8_256_from_frozen_g4_16.json'; v8=asset_dir/'binary_cube_v8_256.json'
    new(k8,dict(schema='trajkv.round8.untrained-cartesian-initialization.v1',centers=product.tolist(),
        parent_book_sha256=sha(old_path),parent_book=base['direction_book_path'],
        codeword_order='j=(first4_index)+(second4_index<<4)',training_performed=False))
    new(v8,dict(schema='trajkv.round8.untrained-binary-cube-initialization.v1',centers=cube.tolist(),
        codeword_order='sum(channel_bit[k]*2**k), k=0..7',training_performed=False))
    assets={p.name:dict(local=str(p),remote=global_root+'/'+p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
            for p in (old_copy,k8,v8)}
    select_path=ROOT/'configs/a2/round8_v1/calibration_select32.json'
    selected=copy.deepcopy(read(select_path)[:2]); assert len({row['source_id'] for row in selected})==2
    for i,row in enumerate(selected):
        row['id']=row['session_id']='r8_identity_select_source_'+str(i+1)
        row['questions']=row['questions'][:2]; row['turns']=len(row['questions'])
        row['source_question_ids']=row['source_question_ids'][:2]
        row['content']=[{'type':'image'},{'type':'text','text':row['questions'][0]}]
        row['round8_analysis_role']='historically_exposed_selection_identity_smoke_not_confirmatory'
        row['split']='round8_calSelect_first2_source_max2Q_identity_control'
        assert not any(key in row for key in ('answer','answers','references'))
    sample_path=ROOT/'configs/a2/round8_v1/identity_select2_sources_max2q.json'; new(sample_path,selected)
    baseline=copy.deepcopy(base); baseline['name']='r7_frozen_joint43_control'
    r8=copy.deepcopy(base); r8.update(name='r8_oldK_binaryV_identity',method='round8_joint',
        direction_book_path=assets[old_copy.name]['remote'],metadata_reference_book_path=base['direction_book_path'],
        round8_factor_policy='inherit_from_reference_book',key_assignment='l2',key_encoding='reencode',value_encoding='binary')
    variants=[baseline,r8]
    for name,key8,value8,joint in [('r8_cartesianK_binaryV_identity',True,False,False),
            ('r8_oldK_cubeV_identity',False,True,False),('r8_cartesianK_cubeV_identity',True,True,False),
            ('r8_oldK_binaryV_jointgain_control',False,False,True)]:
        variant=copy.deepcopy(r8); variant['name']=name
        if key8: variant['direction_book_path']=assets[k8.name]['remote']
        if value8: variant.update(value_encoding='cube-identity',value_joint_book_path=assets[v8.name]['remote'])
        if joint: variant['joint_gain_refit']=True
        variants.append(variant)
    config=copy.deepcopy(original); config.update(start=0,count=2,variants=variants)
    config['arguments']['samples']=stage_root+'/'+sample_path.relative_to(ROOT).as_posix()
    config['arguments']['turns']=2
    cp=ROOT/'configs/a2/round8_v1/identity_smoke6.json'; new(cp,config)
    job=read(ROOT/'configs/a2/round7_v1/job_key_smoke2.json')
    job.update(name='round8_identity_smoke6_select2_v1',cwd=stage_root)
    job['argv']=[job['argv'][0],stage_root+'/code/a2/round8_runner.py','--config',stage_root+'/'+cp.relative_to(ROOT).as_posix(),
                 '--output',stage_root+'/results/round8_v1/identity_smoke6_select2_v1']
    job['env'].update({key:stage_root+'/cache/runtime/'+leaf for key,leaf in
        [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]})
    jp=ROOT/'configs/a2/round8_v1/job_identity_smoke6_select2_v1.json'; new(jp,job)
    budgets=[]
    for nv in (768,1024,1280,1792):
        for variant in variants:
            key8=variant.get('direction_book_path')==assets[k8.name]['remote']; value8=variant.get('value_encoding')=='cube-identity'
            row=cost(nv,28,4,8 if key8 else 4,256 if key8 else 16,8 if value8 else None,256 if value8 else None,
                stages=43,ranking_entries=224)
            row['variant']=variant['name']; budgets.append(row)
    audit=dict(schema='trajkv.round8.identity-smoke-preparation.v1',status='PREPARED_NOT_EXECUTED',
        created_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),
        config=str(cp.relative_to(ROOT)),config_sha256=sha(cp),job=str(jp.relative_to(ROOT)),job_sha256=sha(jp),
        samples=str(sample_path.relative_to(ROOT)),samples_sha256=sha(sample_path),sources=2,
        questions=sum(row['turns'] for row in selected),variants=len(variants),expected_sessions=2*len(variants),
        expected_answers=sum(row['turns'] for row in selected)*len(variants),assets=assets,
        frozen_baseline_config_sha256=sha(original_path),baseline_variant_exact_except_display_name=True,
        requested_ranking_entries=224,actual_stage_limit=43,requested_priority_sha256=hashlib.sha256(json.dumps(base['selected_stages'],separators=(',',':')).encode()).hexdigest(),
        raw_sample_selects='first two stored calSelect32 sources, each at most two questions; no result filtering',
        fit_or_training_performed=False,confirmation_answers_opened=False,GPU_used=False,uploaded=False,
        execution_scope='Free generation smoke only. Tensor/read equivalence must be checked separately; Cartesian membership alone does not prove FP32 tie/rounding identity.',
        root_scheduling='Root may edit an unfrozen dispatch job GPU id or stage paths; this preparer does not connect or launch',budgets=budgets)
    new(ROOT/'audit/a2/round8_identity_smoke_preparation_v1.json',audit)
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',variants=len(variants),sources=2,questions_per_method=audit['questions'],
        config=audit['config'],job=audit['job'],baseline_N1792_bytes=next(x['complete_visual_owner_bytes'] for x in budgets if x['visual_tokens']==1792),GPU_used=False,uploaded=False)))


if __name__=='__main__': main()
