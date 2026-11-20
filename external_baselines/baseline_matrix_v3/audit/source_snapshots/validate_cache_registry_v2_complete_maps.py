"""CPU validation of bounded registry admission and unchanged legacy reuse."""
import ast
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import prediction_cache as pc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='v2');args=parser.parse_args()
    assert args.run_id.replace('_','').isalnum()
    work=Path('${EVIDENCE_ROOT}/cache_validation')/('cache_registry_'+args.run_id);work.mkdir(exist_ok=False)
    policy=pc.read(ROOT/'configs/cache/evaluate_identity_equivalence_v2.json')
    v1hash=pc.sha(ROOT/'configs/cache/evaluate_identity_equivalence_v1.json')
    require_v1='05a456a447c3d9fd7eac14d6cc1d2b0d98abcf5c67dfa3c48ec0268f95673703'
    assert v1hash==require_v1
    cases=[]
    def passed(name,**details):cases.append({'case':name,'status':'PASS',**details})
    def refused(name,call):
        try:call()
        except pc.CacheIntegrityError as exc:passed(name,error=str(exc))
        else:raise AssertionError('Unsafe acceptance: '+name)
    before=ROOT/'audit/source_snapshots/evaluate_stat_binding_before_registry_v2.py'
    after=ROOT/'code/evaluate.py'
    oldtree,newtree=ast.parse(before.read_bytes()),ast.parse(after.read_bytes())
    find=lambda tree:next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOWLIST' for t in n.targets))
    oldlist,newlist=find(oldtree),find(newtree)
    assert set(ast.literal_eval(newlist.value))==set(pc.LEGACY_ADAPTER_PAIRS)|set(pc.APPENDED_ADAPTER_PAIRS)
    newlist.value=oldlist.value
    assert ast.dump(oldtree,include_attributes=False)==ast.dump(newtree,include_attributes=False)
    passed('evaluate_only_exact_three_tuple_append_all_other_AST_unchanged',before_sha256=pc.sha(before),after_sha256=pc.sha(after))
    semantics=[]
    for digest,item in policy['allowed_sources'].items():
        path=ROOT/item['archived_source'];assert pc.sha(path)==digest
        semantics.append(pc.evaluate_semantics(path,digest))
    assert semantics[0]==semantics[1]==semantics[2]
    passed('all_three_explicit_sources_have_same_legacy_numerical_key',key=semantics[0])
    text=after.read_text(encoding='utf-8')
    removed=text.replace('    ("methods.kivi_style", "KIVIStyleAdapter"),\n','')
    refused('legacy_pair_removal_refused',lambda:pc.evaluate_ast_registry_identity(removed.encode()))
    extra=text.replace('    ("methods.eviction_adapter", "PyramidKVVisualAdapter"),',
        '    ("methods.eviction_adapter", "PyramidKVVisualAdapter"),\n    ("methods.unapproved", "UnknownAdapter"),')
    refused('unapproved_fourth_pair_refused',lambda:pc.evaluate_ast_registry_identity(extra.encode()))
    changed=text.replace('torch.manual_seed(0)','torch.manual_seed(1)')
    assert pc.evaluate_ast_registry_identity(changed.encode())[1]!=policy['numerical_program_ast_sha256']
    passed('numerical_statement_change_changes_AST_key')
    for name,body in [('unknown_comment_only_version',text+'\n# unreviewed whole source version\n'),
                      ('changed_numerical_source',changed),
                      ('unknown_identity_body',text.replace('Small processor/config files are always freshly hashed.','UNREVIEWED identity body.'))]:
        file=work/(name+'.py');file.write_text(body,encoding='utf-8')
        assert file.read_bytes()!=after.read_bytes()
        refused(name+'_refused',lambda file=file:pc.evaluate_semantics(file,pc.sha(file)))
    legacy_plans={}
    for tag in ('F','K2_R0','C1_bias32','E_r0888'):
        raw=Path('${EVIDENCE_ROOT}/results/p1')/f'M1_{tag}_textvqa_p1_128_v1'
        original=pc.read(ROOT/f'configs/eval/M1_{tag}_textvqa_p2_500_v1.json')
        plan,_=pc.plan_reuse(original,ROOT/'configs/datasets/v3/textvqa_p2_inputs.jsonl',raw,[raw])
        frozen=pc.read(Path('${EVIDENCE_ROOT}/results/cache_plans/p2_text_info_v1')/(f'M1_{tag}_textvqa_p2_500_v1.json'))
        assert plan['cached_PASS_QA']==128 and plan['missing_QA']==372
        assert plan['effective_contract_sha256']==frozen['effective_contract_sha256']
        legacy_plans[tag]={'hits':128,'missing':372,'contract_sha256':plan['effective_contract_sha256']}
        passed('unchanged_'+tag+'_P1_to_P2_reuse',**legacy_plans[tag])
    reference=pc.SourceRun(Path('${EVIDENCE_ROOT}/results/p1/M1_K2_R0_textvqa_p1_128_v1'))
    target=pc.read(ROOT/'configs/eval/M1_K2_R0_textvqa_p2_500_v1.json')
    for name,mutate in [('cap',lambda c:c['preprocessing'].update(max_patches=4)),
                        ('constructor',lambda c:c['method']['constructor_args'].update(bits=4)),
                        ('prompt',lambda c:c.update(question_suffix='\nDifferent instruction.'))]:
        cfg=deepcopy(target);mutate(cfg)
        refused('changed_'+name+'_still_invalidates',lambda cfg=cfg:pc.plan_reuse(cfg,ROOT/'configs/datasets/v3/textvqa_p2_inputs.jsonl',reference,[reference.path]))
    sourcecopy=work/'source_fault';sourcecopy.mkdir()
    for name in reference.contract['relevant_source_hashes']:
        p=sourcecopy/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    changed=sourcecopy/'code/methods/kivi_reader.py'
    with changed.open('ab') as stream:stream.write(b'\n# CPU source-change fault\n')
    refused('changed_executed_reader_still_invalidates',lambda:pc.plan_reuse(target,ROOT/'configs/datasets/v3/textvqa_p2_inputs.jsonl',reference,[reference.path],source_root=sourcecopy))
    # New-pair dependency-map checks are in-memory structural controls. They
    # are NOT fabricated raw runs or CUDA/numerical acceptance evidence.
    current_inventory=pc.read(reference.path/'source_inventory.json')
    for path in ROOT.glob('code/methods/*.py'):current_inventory[path.relative_to(ROOT).as_posix()]=pc.sha(path)
    for path in ROOT.glob('configs/methods/*.json'):current_inventory[path.relative_to(ROOT).as_posix()]=pc.sha(path)
    current_inventory['code/evaluate.py']=pc.sha(after)
    current_inventory['third_party/kivi/example.py']=pc.sha(ROOT/'third_party/kivi/example.py')
    base=deepcopy(reference.manifest)
    base['source_hashes']={name:pc.sha(ROOT/name) for name in base['source_hashes']}
    base['method_and_reader_source_hashes']={p.name:pc.sha(p) for p in ROOT.glob('code/methods/*.py')}
    deps={}
    for module,cls in pc.APPENDED_ADAPTER_PAIRS:
        fake=deepcopy(base);fake['config']['method'].update(adapter_module=module,adapter_class=cls)
        fake['method_source_sha256']=pc.sha(ROOT/('code/'+module.replace('.','/')+'.py'))
        mapped=pc._dependencies(fake,current_inventory,after);deps[cls]=mapped
        if cls=='KIVIExampleAdapter':assert 'code/methods/kivi_example_reader.py' in mapped and 'third_party/kivi/quant/new_pack.py' in mapped
        elif cls=='TurboQuantBufferedAdapter':assert 'code/methods/turboquant_buffered_adapter.py' in mapped and 'code/methods/turboquant_adapter.py' in mapped
        else:
            assert 'third_party/turboquant_onlyterp/src/cache.py' in mapped and 'configs/methods/onlyterp_mse_source_lock_v2.json' in mapped
            assert not any(name.endswith(('src/kernels.py','src/lut_attention.py','vllm_integration.py','LICENSE')) for name in mapped)
        passed(cls+'_has_independent_executed_dependencies',dependency_files=len(mapped))
    oldT=pc.SourceRun(Path('${EVIDENCE_ROOT}/results/p1/M1_T_K3V2_textvqa_p1_128_v1'))
    cfg=deepcopy(oldT.manifest['config']);cfg['method'].update(adapter_module='methods.turboquant_buffered_adapter',adapter_class='TurboQuantBufferedAdapter')
    cfg['method']['constructor_args']['recent_window']=128
    assert pc.normalize_behavior(cfg,oldT.manifest['model_spec'])!=oldT.contract['behavior']
    assert deps['TurboQuantBufferedAdapter']!=oldT.contract['relevant_source_hashes']
    passed('buffered_class_constructor_source_cannot_hit_bad_T0_contract')
    fake=deepcopy(base);fake['config']['method'].update(adapter_module='methods.turboquant_onlyterp_mse_adapter',adapter_class='OnlyTerpMSEAdapter')
    altered_inventory=deepcopy(current_inventory);altered_inventory['code/methods/turboquant_onlyterp_mse_adapter.py']='0'*64
    refused('OnlyTerp_unreviewed_module_revision_rejected',lambda:pc._dependencies(fake,altered_inventory,after))
    assert deps['OnlyTerpMSEAdapter']!=oldT.contract['relevant_source_hashes']
    passed('OnlyTerp_self_contained_MSE_dependencies_cannot_hit_bad_T0_contract')
    assert pc.sha(ROOT/'configs/cache/evaluate_identity_equivalence_v1.json')==v1hash
    passed('old_policy_v1_exact_bytes_preserved',sha256=v1hash)
    report={'status':'PASS_CPU_BOUNDED_REGISTRY_AND_LEGACY_CACHE_COMPATIBILITY','GPU_used':False,'new_model_inference_QA':0,
        'cases':cases,'case_count':len(cases),'legacy_reuse':legacy_plans,
        'evaluate_source_sha256':pc.sha(after),'cache_engine_sha256':pc.sha(ROOT/'code/prediction_cache.py'),
        'policy_v2_sha256':pc.sha(ROOT/'configs/cache/evaluate_identity_equivalence_v2.json'),
        'prior_policy_v1_sha256':v1hash,'new_port_CUDA_or_main_readiness_claimed':False,
        'OnlyTerp_dependency_status':'CONFIRMED_THREE_RUNTIME_FILES_CPU_V2_SOURCE; CUDA_AND_MAIN_PENDING',
        'new_map_controls':'CPU in-memory metadata only; not produced GPU model outputs'}
    p=ROOT/'audit/cache_registry_v2_complete_maps_acceptance.json'
    with p.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'status':report['status'],'cases':len(cases),'new_GPU_QA':0}))


if __name__=='__main__':main()
