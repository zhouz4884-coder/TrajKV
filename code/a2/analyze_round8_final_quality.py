"""Final-marker-only wrapper for Mean plus all preserved R8 quality families.

The original scorer and completed-family guard stay byte-identical. Every
producer is gated before labels are opened; unfinished Mean records are never
read. Explicit controls can be supplied as separate original completed runs.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import analyze_round8_completed_quality as previous

base = previous.base


def record_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def record_identity(row):
    sid = row.get('session_id',row.get('id'))
    base.require(isinstance(sid,str) and sid and ('id' not in row or row['id'] == sid),
                 'Missing/inconsistent Mean producer session identity')
    return sid


def mean_owner_contract(owner):
    """Validate the actual owned() mapping without adding text to visual cost."""
    byte_fields = ('complete_visual_owner_bytes','visual_fp16_denominator','visual_base','visual_residual',
        'selector_metadata','visual_position','encoder_constant_bytes','text_capacity','text_position_capacity',
        'position','total_unique','diagnostic_fp16_reference','kvarn_packed_reference')
    return (isinstance(owner,dict) and all(type(owner.get(name)) is int and owner[name] >= 0 for name in byte_fields)
        and all(owner[name] > 0 for name in ('complete_visual_owner_bytes','visual_fp16_denominator','visual_base'))
        and owner['complete_visual_owner_bytes'] == sum(owner[name] for name in
            ('visual_base','visual_residual','selector_metadata','visual_position','encoder_constant_bytes'))
        and owner['position'] == owner['visual_position']+owner['text_position_capacity']
        and owner['total_unique'] >= owner['complete_visual_owner_bytes']
        and owner.get('full_visual_retained') is False
        and owner['diagnostic_fp16_reference'] == owner['kvarn_packed_reference'] == 0)


def completion_gate(folder, read_record_rows=None):
    manifest_path = folder/'manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = base.read(manifest_path)
    family = manifest.get('backend_family') or manifest.get('config', {}).get('backend_family')
    if family != 'mean':
        base.require(not (folder/'complete_mean.json').is_file(), 'Mean final marker cannot be used by another backend')
        return previous.completion_gate(folder)
    marker_path = folder/'complete_mean.json'
    if not marker_path.is_file():
        return dict(status='NOT_COMPLETE_NO_SCORING', run=str(folder), reason='final complete_mean.json pending',
                    references_opened=False, output_written=False, raw_answers_opened=False)
    require = base.require
    final = base.read(marker_path)
    require(final.get('status') == 'COMPLETE' and final.get('backend_family') == 'mean'
        and final.get('candidate_subset') is True and final.get('factorial_complete') is False
        and final.get('accepted_only_after_this_marker') is True
        and final.get('added_persistent_tensor_bytes') == 0, 'Invalid Mean final completion contract')
    hashes = {name:base.sha(folder/name) for name in ('complete_round8.json','manifest.json','records.jsonl','complete.json')}
    for key, name in (('complete_round8_sha256','complete_round8.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')):
        require(final.get(key) == hashes[name], 'Mean final marker canonical SHA mismatch: '+name)
    sources = final.get('source_sha256', {})
    runner_name = final.get('runner_source_name','round8_mean_runner.py')
    require(runner_name in ('round8_mean_runner.py','round8_mean_runner_v2.py'),
            'Unrecognized Mean runner source name')
    require(isinstance(sources, dict) and sources and all(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value)
        for value in sources.values()) and final.get('runner_sha256')
        and final.get('runner_sha256') == sources.get(runner_name),
        'Mean runner/source binding mismatch')
    require(isinstance(final.get('config_sha256'),str) and re.fullmatch('[0-9a-f]{64}',final['config_sha256'])
        and final.get('config_sha256') == manifest.get('config_sha256')
        and manifest.get('backend_family') == 'mean' and manifest['config'].get('backend_family') == 'mean'
        and manifest.get('mean_source_sha256') == sources
        and manifest.get('candidate_subset') is True and manifest.get('factorial_complete') is False,
        'Mean manifest source/config/scope mismatch')
    post = base.read(folder/'complete_round8.json')
    require(post.get('status') == 'COMPLETE' and post.get('encoder_inputs_unchanged') is True
        and post.get('source_inputs_unchanged') is True and post.get('backend_family') == 'mean'
        and post.get('candidate_subset') is True and post.get('factorial_complete') is False
        and post.get('mean_source_sha256') == sources, 'Mean intermediate completion contract differs')
    for key, name in (('engine_complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')):
        require(post.get(key) == hashes[name], 'Mean intermediate canonical SHA mismatch: '+name)
    engine = base.read(folder/'complete.json')
    require(engine.get('status') == 'COMPLETE' and engine.get('records') == engine.get('expected')
        and engine.get('expected',0) > 0, 'Mean engine coverage is incomplete')
    from round8_mean_runner import validate_config
    declarations = json.loads(json.dumps(validate_config(manifest['config'])))
    names = [variant['name'] for variant in manifest['config']['variants']]
    require(manifest.get('mean_declarations') == declarations and manifest.get('executed_candidate_variants') == names,
        'Mean executed declarations differ from original config')
    external = final.get('external_inputs_sha256', {})
    require(isinstance(external,dict) and external and all(manifest.get('external_codec_inputs',{}).get(name) == value
        for name,value in external.items()) and manifest.get('samples_sha256') == external.get(manifest['config']['arguments']['samples']),
        'Mean frozen external/sample input graph differs')
    broad = manifest.get('sources', {})
    if 'code/a2/'+runner_name in broad:
        require(broad['code/a2/'+runner_name] == final['runner_sha256'], 'Mean broad runner inventory differs')
    # Only a valid, hash-bound final marker permits reading raw answer records.
    rows = (read_record_rows or record_rows)(folder/'records.jsonl')
    expected = {(sid,name) for sid in manifest['selected_ids'] for name in names}
    actual = [(record_identity(row),row['variant']) for row in rows]
    require(len(manifest['selected_ids']) == len(set(manifest['selected_ids'])) and expected
        and len(actual) == len(set(actual)) and set(actual) == expected and len(rows) == engine['records'],
        'Mean selected-source x variant coverage differs')
    variants = {variant['name']:variant for variant in manifest['config']['variants']}
    coverage = []
    for row in rows:
        name = row['variant']; options = declarations[name]
        require(row.get('variant_config') == variants[name] and row.get('method') == variants[name]['method']
            and record_identity(row) in manifest['selected_ids'], 'Mean record variant/session mismatch')
        if options is None:
            continue
        integrity = row.get('mean_header_integrity', {})
        statistics = row.get('mean_correction_statistics', [])
        fields = integrity.get('fields_before', {})
        require(row.get('visual_mean_correction') is options['enabled'] and row.get('mean_source_sha256') == sources
            and row.get('mean_actual_decode_chunk') == options['mean_decode_chunk']
            and row.get('mean_correction_added_persistent_tensor_bytes') == 0
            and row.get('mean_correction_source_FP16_used') is False and row.get('mean_reader_unchanged') is True
            and integrity.get('layer_count') == 28 and integrity.get('non_mean_tensor_fingerprints_and_pointers_unchanged') is True
            and integrity.get('mean_pointers_preserved') is True and integrity.get('actual_owner_unchanged') is True
            and mean_owner_contract(integrity.get('actual_owner_before'))
            and integrity.get('actual_owner_before') == integrity.get('actual_owner_after')
            and fields == integrity.get('fields_after') and len(fields.get('non_mean_layers',[])) == 28
            and len(fields.get('mean_pointers',[])) == 28 and len(fields.get('mean_shapes',[])) == 28
            and len(statistics) == (28 if options['enabled'] else 0), 'Mean unchanged-field/owner/layer evidence differs')
        if options['enabled']:
            require(sorted(item['layer'] for item in statistics) == list(range(28)) and all(
                item['visual_tokens'] == row['visual_tokens'] and item.get('full_visual_mean') is True
                and item.get('source_FP16_used') is False and item.get('mean_storage_pointer_preserved') is True
                and item.get('added_persistent_tensor_bytes') == 0 and item.get('apply_count') == 1
                and item.get('actual_decode_chunk') == options['mean_decode_chunk'] for item in statistics), 'Mean all-real-token coverage differs')
        coverage.append(dict(id=record_identity(row), variant=name, enabled=options['enabled'],
            layers=28, real_visual_tokens=row['visual_tokens']))
    require(final.get('mean_record_coverage') == coverage, 'Mean final coverage receipt differs from original records')
    return dict(status='PASS',run=str(folder),backend_family='mean',final_marker='complete_mean.json',
        final_marker_sha256=base.sha(marker_path),canonical_sha256=hashes,source_sha256=sources,
        runner_source_name=runner_name,runner_sha256=final['runner_sha256'],config_sha256=final['config_sha256'],variants=names,
        mean_records=len(coverage),candidate_subset=True,factorial_complete=False,
        added_persistent_tensor_bytes=0,reference_content_opened=False)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        return base.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runs',nargs='+',required=True); parser.add_argument('--output',type=Path,required=True)
    args,_ = parser.parse_known_args()
    base.require(not args.output.exists(), 'Use a new immutable quality analysis output')
    receipts = []
    for value in args.runs:
        folder = Path(value)
        if not folder.is_absolute():
            folder = base.ROOT/folder if (base.ROOT/folder).is_dir() else base.ROOT/'results'/folder
        for child in ('quality','transfer32'):
            if (folder/child/'records.jsonl').is_file():
                folder = folder/child; break
        result = completion_gate(folder)
        if result and result['status'] != 'PASS':
            print(json.dumps(result)); return
        if result:
            receipts.append(result)
    base.main()
    summary = args.output/'summary.json'
    if summary.is_file():
        with (args.output/'final_completion_audit.json').open('x',encoding='utf-8') as stream:
            json.dump(dict(status='PASS',final_markers_verified_before_reference_access=True,
                specialized_runs=receipts,summary_sha256=base.sha(summary),guard_sha256=base.sha(__file__),
                inherited_analyzer_sha256=base.sha(Path(base.__file__)),
                previous_guard_sha256=base.sha(Path(previous.__file__)),
                scope='Final completion gates only; inherited scoring, owner, references and statistics unchanged'),stream,indent=2)


if __name__ == '__main__':
    main()
