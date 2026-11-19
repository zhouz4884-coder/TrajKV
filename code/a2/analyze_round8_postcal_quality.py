"""Final-postcal marker/owner gate before inherited R8 task scoring.

Old scorers, completed producers and numerical sources remain byte-identical.
The new owner is used only for explicitly declared postcal R8 records.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import analyze_round8_final_quality as previous

base = previous.base


def completion_gate(folder, read_record_rows=None):
    # Publication verification supplies a read-only virtual archive directory.
    if isinstance(folder, (str, Path)):
        folder = Path(folder)
    path = folder / 'manifest.json'
    if not path.is_file():
        return None
    manifest = base.read(path)
    family = manifest.get('backend_family') or manifest.get('config', {}).get('backend_family')
    if family != 'postcal':
        base.require(not (folder / 'complete_postcal.json').is_file(),
                     'Postcal final marker cannot identify another family')
        return previous.completion_gate(folder, read_record_rows=read_record_rows)
    marker = folder / 'complete_postcal.json'
    if not marker.is_file():
        return dict(status='NOT_COMPLETE_NO_SCORING', run=str(folder),
            reason='final complete_postcal.json pending', references_opened=False,
            raw_answers_opened=False, output_written=False)
    require = base.require
    final = base.read(marker)
    require(final.get('status') == 'COMPLETE' and final.get('backend_family') == 'postcal'
        and final.get('candidate_subset') is True and final.get('factorial_complete') is False
        and final.get('accepted_only_after_this_marker') is True,
        'Invalid postcal final completion contract')
    canonical = {name: base.sha(folder / name) for name in
                 ('complete.json', 'complete_round8.json', 'manifest.json', 'records.jsonl')}
    for key, name in (('complete_round8_sha256', 'complete_round8.json'),
                      ('manifest_sha256', 'manifest.json'), ('records_sha256', 'records.jsonl')):
        require(final.get(key) == canonical[name], 'Postcal final canonical SHA mismatch: ' + name)
    sources = final.get('source_sha256', {})
    require(isinstance(sources, dict) and sources and all(isinstance(value, str)
        and re.fullmatch('[0-9a-f]{64}', value) for value in sources.values())
        and final.get('runner_sha256') == sources.get('round8_postcal_runner.py'),
        'Postcal runner/source binding differs')
    require(final.get('config_sha256') == manifest.get('config_sha256')
        and manifest.get('backend_family') == manifest['config'].get('backend_family') == 'postcal'
        and manifest.get('postcal_source_sha256') == sources
        and manifest.get('candidate_subset') is True and manifest.get('factorial_complete') is False,
        'Postcal manifest source/config/scope differs')
    intermediate = base.read(folder / 'complete_round8.json')
    require(intermediate.get('status') == 'COMPLETE'
        and intermediate.get('encoder_inputs_unchanged') is True
        and intermediate.get('source_inputs_unchanged') is True
        and intermediate.get('backend_family') == 'postcal'
        and intermediate.get('postcal_source_sha256') == sources,
        'Postcal intermediate source/input completion differs')
    for key, name in (('engine_complete_sha256', 'complete.json'),
                      ('manifest_sha256', 'manifest.json'), ('records_sha256', 'records.jsonl')):
        require(intermediate.get(key) == canonical[name], 'Postcal intermediate canonical SHA differs')
    engine = base.read(folder / 'complete.json')
    require(engine.get('status') == 'COMPLETE' and engine.get('records') == engine.get('expected')
        and engine.get('expected', 0) > 0, 'Postcal engine coverage incomplete')
    variants = manifest['config']['variants']
    by_name = {variant['name']: variant for variant in variants}
    require(len(by_name) == len(variants) and manifest.get('executed_candidate_variants') == list(by_name),
        'Postcal executed variant declarations differ')
    # Only the valid, byte-bound final marker permits reading raw answers.
    rows = (read_record_rows or previous.record_rows)(folder / 'records.jsonl')
    ids = manifest['selected_ids']
    actual = [(previous.record_identity(row), row['variant']) for row in rows]
    require(len(ids) == len(set(ids)) and len(actual) == len(set(actual))
        and set(actual) == {(sid, name) for sid in ids for name in by_name}
        and len(actual) == engine['records'], 'Postcal source x variant coverage differs')
    coverage = []
    for row in rows:
        variant = by_name[row['variant']]
        require(row.get('variant_config') == variant and row.get('method') == variant['method'],
                'Postcal record variant configuration differs')
        if row['method'] != 'round8_joint':
            require(variant.get('postcal_mode', 'off') == 'off', 'Non-postcal method enables postcal')
            continue
        mode = variant.get('postcal_mode')
        require(mode in ('identity', 'learned') and row.get('postcal_mode') == mode
            and row.get('postcal_source_sha256') == sources
            and isinstance(row.get('postcal_table_values_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', row['postcal_table_values_sha256'])
            and row.get('postcal_parameter_shape') == [3, 28, 4]
            and row.get('postcal_added_encoder_constant_bytes') == 1344
            and row.get('postcal_added_retained_tensor_bytes') == 1344
            and row.get('postcal_J_sign_mean_Vcode_unchanged') is True
            and row.get('postcal_reader_unchanged') is True
            and row.get('postcal_complete_le_old43') is True,
            'Postcal deployed physical-field/parameter contract differs')
        owner = row['owned_initial']
        old42, old43 = row.get('postcal_original_R42_complete_bytes'), row.get('postcal_old43_complete_bytes')
        require(type(old42) is int and type(old43) is int
            and owner['complete_visual_owner_bytes'] == old42 + 1344
            and old43 == old42 + 18 * row['visual_tokens']
            and owner['complete_visual_owner_bytes'] <= old43,
            'Postcal full actual owner is not R42+1344B within oldR43')
        require(row.get('depth_residual_selected_stages') == variant['selected_stages'][:42]
            and variant['res_stages_limit'] == 42 and len(variant['selected_stages']) == 224,
            'Postcal full ranking/R42 mask differs')
        if mode == 'learned':
            external = manifest.get('external_codec_inputs', {})
            require(row.get('postcal_factor_file_sha256') == external.get(variant['postcal_factor_path'])
                and row.get('postcal_train_complete_sha256') == external.get(variant['postcal_train_complete_path']),
                'Postcal learned factors/training completion not bound to consumed inputs')
        coverage.append(dict(session_id=previous.record_identity(row), variant=row['variant'], mode=mode,
            complete_visual_owner_bytes=owner['complete_visual_owner_bytes'], old43_bytes=old43,
            postcal_table_values_sha256=row.get('postcal_table_values_sha256')))
    return dict(status='PASS', run=str(folder), backend_family='postcal', final_marker='complete_postcal.json',
        final_marker_sha256=base.sha(marker), canonical_sha256=canonical, source_sha256=sources,
        config_sha256=final['config_sha256'], variants=list(by_name),
        postcal_coverage=coverage, candidate_subset=True, factorial_complete=False,
        references_opened=False)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        return base.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    base.require(not args.output.exists(), 'Use a new immutable postcal quality analysis output')
    audits = []
    for value in args.runs:
        folder = Path(value)
        if not folder.is_absolute():
            folder = base.ROOT / folder if (base.ROOT / folder).is_dir() else base.ROOT / 'results' / folder
        for child in ('quality', 'transfer32'):
            if (folder / child / 'records.jsonl').is_file():
                folder = folder / child
                break
        result = completion_gate(folder)
        if result and result['status'] != 'PASS':
            print(json.dumps(result)); return
        if result:
            audits.append(result)
    import round8_postcal_owner as owner
    original_owner = base.audit_owner
    try:
        base.audit_owner = owner.audit_owner
        base.main()
    finally:
        base.audit_owner = original_owner
    summary = args.output / 'summary.json'
    if summary.is_file():
        with (args.output / 'postcal_completion_audit.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(status='PASS', final_markers_verified_before_reference_access=True,
                specialized_runs=audits, summary_sha256=base.sha(summary), guard_sha256=base.sha(__file__),
                new_owner_sha256=base.sha(owner.__file__), inherited_analyzer_sha256=base.sha(base.__file__),
                previous_guard_sha256=base.sha(previous.__file__),
                scope='New postcal completion/owner contracts; inherited task scoring and statistics unchanged'),
                stream, indent=2)


if __name__ == '__main__':
    main()
