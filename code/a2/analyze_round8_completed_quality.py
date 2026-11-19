"""Family-specific final-marker gates before the preserved R8 quality scorer.

Same scoring CLI. Supports dedicated postres/final-ray completion, candidate
spherical/field subsets, and original R8 controls. Labels stay unopened until
every supplied run's declared completion contract has passed.
"""
import argparse
import json
from pathlib import Path
import sys

import analyze_round8_quality as base
from analyze_round8_candidate_quality import candidate_gate

SPECIAL = {
    'postres': ('complete_postres.json', 'round8_postres_runner.py', 'postres_source_sha256'),
    'final_ray': ('complete_final_ray.json', 'round8_final_ray_runner.py', 'final_ray_source_sha256'),
}


def completion_gate(folder):
    path = folder / 'manifest.json'
    if not path.is_file():
        return None
    manifest = base.read(path)
    family = manifest.get('backend_family') or manifest.get('config', {}).get('backend_family')
    if family in ('spherical', 'field'):
        return candidate_gate(folder)
    if family not in SPECIAL:
        base.require(manifest.get('candidate_subset') is not True
            and not any((folder / value[0]).exists() for value in SPECIAL.values()),
            'Unrecognized candidate backend cannot bypass its final completion gate')
        return None
    filename, runner, source_key = SPECIAL[family]
    final_path = folder / filename
    if not final_path.is_file():
        return dict(status='NOT_COMPLETE_NO_SCORING', run=str(folder), reason='final ' + filename + ' pending',
            references_opened=False, output_written=False)
    req = base.require
    final = base.read(final_path)
    req(final.get('status') == 'COMPLETE' and final.get('backend_family') == family
        and final.get('candidate_subset') is True and final.get('factorial_complete') is False
        and final.get('accepted_only_after_this_marker') is True, 'Invalid dedicated final completion contract')
    hashes = {name: base.sha(folder / name) for name in ('complete_round8.json', 'manifest.json', 'records.jsonl', 'complete.json')}
    for field, name in [('complete_round8_sha256', 'complete_round8.json'),
            ('manifest_sha256', 'manifest.json'), ('records_sha256', 'records.jsonl')]:
        req(final.get(field) == hashes[name], 'Dedicated final-marker canonical SHA mismatch: ' + name)
    sources = final.get('source_sha256', {})
    req(final.get('runner_sha256') and final.get('runner_sha256') == sources.get(runner), 'Runner/source SHA binding mismatch')
    req(final.get('config_sha256') == manifest.get('config_sha256')
        and manifest.get('backend_family') == family and manifest['config'].get('backend_family') == family,
        'Dedicated family/config identity mismatch')
    req(manifest.get('candidate_subset') is True and manifest.get('factorial_complete') is False
        and manifest.get(source_key) == sources, 'Dedicated manifest source/scope mismatch')
    marker = base.read(folder / 'complete_round8.json')
    req(marker.get('status') == 'COMPLETE' and marker.get('encoder_inputs_unchanged') is True
        and marker.get('source_inputs_unchanged') is True and marker.get('backend_family') == family
        and marker.get('candidate_subset') is True and marker.get('factorial_complete') is False
        and marker.get(source_key) == sources, 'Intermediate R8 marker source/scope mismatch')
    for field, name in [('engine_complete_sha256', 'complete.json'),
            ('manifest_sha256', 'manifest.json'), ('records_sha256', 'records.jsonl')]:
        req(marker.get(field) == hashes[name], 'R8 canonical file differs from dedicated final marker')
    engine = base.read(folder / 'complete.json')
    req(engine.get('status') == 'COMPLETE' and engine.get('records') == engine.get('expected')
        and engine.get('expected', 0) > 0, 'Incomplete engine record coverage')
    names = [variant['name'] for variant in manifest['config']['variants']]
    req(manifest.get('executed_candidate_variants') == names, 'Executed variant list differs from producer config')
    broad_sources = manifest.get('sources', {})
    if isinstance(broad_sources, dict) and 'code/a2/' + runner in broad_sources:
        req(broad_sources['code/a2/' + runner] == final['runner_sha256'], 'Broad producer inventory and specialized runner source differ')
    return dict(status='PASS', run=str(folder), backend_family=family, final_marker=filename,
        final_marker_sha256=base.sha(final_path), canonical_sha256=hashes, runner_sha256=final['runner_sha256'],
        source_sha256=sources, config_sha256=final['config_sha256'], variants=names,
        candidate_subset=True, factorial_complete=False)


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        return base.main()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    base.require(not args.output.exists(), 'Use a new analysis output; preserve existing summaries and audits')
    receipts = []
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
            print(json.dumps(result))
            return
        if result:
            receipts.append(result)
    base.main()
    summary = args.output / 'summary.json'
    if summary.is_file():
        with (args.output / 'specialized_completion_audit.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(status='PASS', final_markers_verified_before_reference_access=True,
                specialized_runs=receipts, summary_sha256=base.sha(summary), guard_sha256=base.sha(__file__),
                inherited_analyzer_sha256=base.sha(Path(base.__file__)),
                scope='Dedicated completion gates only; original scoring, owners, paired statistics and reference bytes unchanged'),
                stream, indent=2)


if __name__ == '__main__':
    main()
