"""CPU metadata fixtures for final Mean gates; no model, GPU or labels."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import analyze_round8_final_quality as gate
from round8_mean_runner import validate_config

sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def fixture(folder,change=None,runner_name='round8_mean_runner.py',session_id_only=False):
    folder.mkdir(parents=True)
    stages = [[axis,layer,head,0] for axis in ('k','v') for layer in range(28) for head in range(4)]
    variants = [dict(name='full',method='full')]+[dict(name='mean_'+str(enabled),method='round8_joint',
        visual_mean_correction=enabled,selected_stages=stages,res_stages_limit=43,
        key_assignment='l2',key_amplitude='norm',key_encoding='reencode',value_encoding='binary',
        v_group_size='all',joint_gain_refit=False,value_joint_refit=False,cap=8,budget_bits=1.8,
        direction_book_path='old_book.json',metadata_reference_book_path='old_book.json',
        metadata_factor_path='factors.json',round8_factor_policy='frozen') for enabled in (False,True)]
    config = dict(backend_family='mean',arguments=dict(samples='samples.json'),variants=variants)
    declarations = json.loads(json.dumps(validate_config(config)))
    sources = {runner_name:'a'*64,'round8_mean_correction.py':'b'*64}
    external = {'samples.json':'c'*64,'old_book.json':'d'*64}
    rows,coverage = [],[]
    for variant in variants:
        row = dict(id='sample1',session_id='sample1',variant=variant['name'],method=variant['method'],
            variant_config=variant,visual_tokens=129)
        if session_id_only:
            row.pop('id')
        option = declarations[variant['name']]
        if option is not None:
            owner = dict(complete_visual_owner_bytes=1264,visual_fp16_denominator=16384,visual_base=1024,
                visual_residual=128,selector_metadata=16,visual_position=64,encoder_constant_bytes=32,
                text_capacity=256,text_position_capacity=16,position=80,total_unique=1536,
                diagnostic_fp16_reference=0,kvarn_packed_reference=0,full_visual_retained=False)
            fields = dict(book={'sha256':'e'*64},non_mean_layers=[{'synthetic_layer':layer} for layer in range(28)],
                mean_pointers=list(range(28)),mean_shapes=[[1,4,1,128] for _ in range(28)])
            row.update(visual_mean_correction=option['enabled'],mean_source_sha256=sources,
                mean_actual_decode_chunk=option['mean_decode_chunk'],
                mean_correction_added_persistent_tensor_bytes=0,mean_correction_source_FP16_used=False,
                mean_reader_unchanged=True,mean_header_integrity=dict(layer_count=28,
                    non_mean_tensor_fingerprints_and_pointers_unchanged=True,mean_pointers_preserved=True,
                    actual_owner_unchanged=True,actual_owner_before=owner,actual_owner_after=deepcopy(owner),
                    fields_before=fields,fields_after=deepcopy(fields)),
                mean_correction_statistics=[dict(layer=layer,visual_tokens=129,full_visual_mean=True,
                    source_FP16_used=False,mean_storage_pointer_preserved=True,added_persistent_tensor_bytes=0,
                    apply_count=1,actual_decode_chunk=option['mean_decode_chunk'])
                    for layer in range(28)] if option['enabled'] else [])
            coverage.append(dict(id='sample1',variant=variant['name'],enabled=option['enabled'],layers=28,real_visual_tokens=129))
        rows.append(row)
    manifest = dict(config=config,config_sha256='f'*64,selected_ids=['sample1'],samples_sha256=external['samples.json'],
        external_codec_inputs=external,backend_family='mean',candidate_subset=True,factorial_complete=False,
        mean_declarations=declarations,mean_source_sha256=sources,executed_candidate_variants=list(declarations),
        sources={'code/a2/'+runner_name:sources[runner_name]})
    engine = dict(status='COMPLETE',records=3,expected=3)
    if change:
        change(manifest,engine,rows,coverage)
    write(folder/'manifest.json',manifest); write(folder/'complete.json',engine)
    (folder/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    post = dict(status='COMPLETE',engine_complete_sha256=sha(folder/'complete.json'),manifest_sha256=sha(folder/'manifest.json'),
        records_sha256=sha(folder/'records.jsonl'),encoder_inputs_unchanged=True,source_inputs_unchanged=True,
        backend_family='mean',candidate_subset=True,factorial_complete=False,mean_source_sha256=sources)
    write(folder/'complete_round8.json',post)
    final = dict(status='COMPLETE',complete_round8_sha256=sha(folder/'complete_round8.json'),
        manifest_sha256=sha(folder/'manifest.json'),records_sha256=sha(folder/'records.jsonl'),
        config_sha256=manifest['config_sha256'],runner_sha256=sources[runner_name],source_sha256=sources,
        external_inputs_sha256=external,backend_family='mean',candidate_subset=True,factorial_complete=False,
        mean_record_coverage=coverage,added_persistent_tensor_bytes=0,accepted_only_after_this_marker=True)
    if runner_name != 'round8_mean_runner.py':
        final['runner_source_name'] = runner_name
    write(folder/'complete_mean.json',final)
    return folder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new fixture directory')
    args.output.mkdir(parents=True)
    tests=[]
    valid=fixture(args.output/'valid')
    assert gate.completion_gate(valid)['status']=='PASS'
    tests.append(dict(case='complete_bound_mean_off_on_with_full_control',result='PASS'))
    valid_v2=fixture(args.output/'valid_v2_session_id_only',runner_name='round8_mean_runner_v2.py',session_id_only=True)
    assert gate.completion_gate(valid_v2)['runner_source_name']=='round8_mean_runner_v2.py'
    tests.append(dict(case='complete_v2_session_id_only_with_full_control',result='PASS'))
    pending=fixture(args.output/'pending');(pending/'complete_mean.json').unlink()
    def unopened(_):raise AssertionError('Unfinished raw answers were opened')
    result=gate.completion_gate(pending,read_record_rows=unopened)
    assert result['status']=='NOT_COMPLETE_NO_SCORING' and result['raw_answers_opened'] is False
    tests.append(dict(case='missing_final_marker_before_raw_answers',result='REJECTED_NO_SCORING'))
    changes = {
        'partial_engine':lambda m,e,r,c:e.update(records=2),
        'duplicate_record':lambda m,e,r,c:r.__setitem__(2,deepcopy(r[1])),
        'wrong_source_density_full_Nv':lambda m,e,r,c:r[2]['mean_correction_statistics'][0].update(visual_tokens=128),
        'missing_layer':lambda m,e,r,c:r[2]['mean_header_integrity'].update(layer_count=27),
        'changed_owner':lambda m,e,r,c:r[2]['mean_header_integrity'].update(actual_owner_after=12346),
        'changed_nonmean_payload':lambda m,e,r,c:r[2]['mean_header_integrity']['fields_after']['non_mean_layers'][0].update(changed=True),
        'used_source_FP16':lambda m,e,r,c:r[2].update(mean_correction_source_FP16_used=True),
        'final_coverage_lie':lambda m,e,r,c:c[-1].update(real_visual_tokens=128),
        'invalid_ranking_math':lambda m,e,r,c:m['config']['variants'][2].update(res_stages_limit=42),
        'ignored_decode_chunk':lambda m,e,r,c:r[2].update(mean_actual_decode_chunk=256),
        'repeated_apply':lambda m,e,r,c:r[2]['mean_correction_statistics'][0].update(apply_count=2),
        'mismatched_id_session_id':lambda m,e,r,c:r[2].update(id='different_sample'),
        'broad_runner_source_mismatch':lambda m,e,r,c:m['sources'].update({'code/a2/round8_mean_runner.py':'e'*64}),
        'integer_owner_misdeclared':lambda m,e,r,c:r[2]['mean_header_integrity'].update(actual_owner_before=12345,actual_owner_after=12345),
        'boolean_owner_byte_field':lambda m,e,r,c:[r[2]['mean_header_integrity'][k].update(visual_base=True) for k in ('actual_owner_before','actual_owner_after')],
        'owner_visual_sum_mismatch':lambda m,e,r,c:[r[2]['mean_header_integrity'][k].update(complete_visual_owner_bytes=1265) for k in ('actual_owner_before','actual_owner_after')],
    }
    for name,change in changes.items():
        folder=fixture(args.output/name,change)
        try:gate.completion_gate(folder)
        except (ValueError,KeyError):tests.append(dict(case=name,result='REJECTED'))
        else:raise AssertionError('Invalid fixture accepted: '+name)
    unknown=fixture(args.output/'unknown_runner',runner_name='unapproved_runner.py',session_id_only=True)
    try:gate.completion_gate(unknown,read_record_rows=unopened)
    except ValueError:tests.append(dict(case='unknown_runner_name_before_raw_answers',result='REJECTED'))
    else:raise AssertionError('Unapproved runner accepted')
    stale=fixture(args.output/'stale_record_digest')
    with (stale/'records.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
    try:gate.completion_gate(stale,read_record_rows=unopened)
    except ValueError:tests.append(dict(case='changed_record_digest_before_raw_answers',result='REJECTED'))
    else:raise AssertionError('Stale record digest accepted')
    result=dict(status='PASS',tests=tests,CPU_only=True,GPU_used=False,model_used=False,references_opened=False,
        task_quality_evidence=False,fixture_scope='Synthetic completion/source/input/coverage and unchanged-field contract tests only',
        gate_sha256=sha(Path(gate.__file__)),fixture_checker_sha256=sha(Path(__file__)))
    write(args.output/'CHECK.json',result);print(json.dumps(dict(status='PASS',tests=len(tests),output=str(args.output))))


if __name__=='__main__':main()
