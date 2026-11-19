"""Strict candidate completion gate before the preserved R8 quality analyzer.

Same CLI as analyze_round8_quality.py. Final candidate marker must bind every
canonical producer file; before_candidate audit copies never count as runs.
No labels are opened by this gate and no existing analysis is rewritten.
"""
import argparse
import json
from pathlib import Path
import sys

import analyze_round8_quality as base


def candidate_gate(folder):
    manifest_path=folder/'manifest.json'
    if not manifest_path.is_file(): return None
    manifest=base.read(manifest_path)
    family=manifest.get('backend_family') or manifest.get('config',{}).get('backend_family')
    is_candidate=(family in ('spherical','field') or manifest.get('candidate_subset') is True
        or (folder/'complete_candidate.json').exists())
    if not is_candidate: return None
    candidate_path=folder/'complete_candidate.json'
    if not candidate_path.is_file():
        return dict(status='NOT_COMPLETE_NO_SCORING',run=str(folder),reason='final complete_candidate.json pending',references_opened=False,output_written=False)
    candidate=base.read(candidate_path); req=base.require
    req(candidate.get('status')=='COMPLETE' and candidate.get('candidate_subset') is True
        and candidate.get('factorial_complete') is False and candidate.get('accepted_only_after_this_marker') is True,
        'Invalid final candidate completion contract')
    req(candidate.get('backend_family')==family and family in ('spherical','field')
        and manifest.get('backend_family')==family and manifest['config'].get('backend_family')==family,
        'Candidate backend-family identity mismatch')
    hashes={name:base.sha(folder/name) for name in ('complete_round8.json','manifest.json','records.jsonl','complete.json','candidate_subset_postcheck.json')}
    for field,name in [('complete_round8_sha256','complete_round8.json'),('manifest_sha256','manifest.json'),
            ('records_sha256','records.jsonl'),('candidate_subset_postcheck_sha256','candidate_subset_postcheck.json')]:
        req(candidate.get(field)==hashes[name],'Candidate canonical file SHA mismatch: '+name)
    req(candidate.get('config_sha256')==manifest.get('config_sha256'),'Candidate config SHA differs from producer manifest')
    sources=candidate.get('source_sha256',{}); wrapper=candidate.get('wrapper_sha256')
    req(wrapper and sources.get('round8_candidate_runner.py')==wrapper
        and manifest.get('candidate_wrapper_sha256')==wrapper,'Candidate wrapper source binding mismatch')
    source_key='spherical_joint_source_sha256' if family=='spherical' else 'field_joint_source_sha256'
    marker=base.read(folder/'complete_round8.json'); post=base.read(folder/'candidate_subset_postcheck.json')
    req(marker.get('status')=='COMPLETE' and marker.get('encoder_inputs_unchanged') is True
        and marker.get('source_inputs_unchanged') is True and marker.get('candidate_subset') is True
        and marker.get('factorial_complete') is False and marker.get('backend_family')==family,
        'Candidate backend postcheck incomplete')
    req(marker.get('engine_complete_sha256')==hashes['complete.json']
        and marker.get('manifest_sha256')==hashes['manifest.json']
        and marker.get('records_sha256')==hashes['records.jsonl']
        and marker.get('candidate_subset_postcheck_sha256')==hashes['candidate_subset_postcheck.json']
        and marker.get('candidate_wrapper_sha256')==wrapper,'Candidate backend canonical hashes differ')
    req(marker.get(source_key)==sources and manifest.get(source_key)==sources,'Candidate family source mapping differs')
    req(post.get('status')=='COMPLETE' and post.get('backend_family')==family
        and post.get('candidate_subset') is True and post.get('factorial_complete') is False
        and post.get('source_sha256')==sources,'Candidate subset postcheck contract differs')
    for field,name in [('engine_complete_sha256','complete.json'),('manifest_sha256','manifest.json'),('records_sha256','records.jsonl')]:
        req(post.get(field)==hashes[name],'Subset postcheck canonical SHA mismatch')
    for field,name in [('backend_manifest_before_metadata_sha256','backend_manifest_before_candidate.json'),
            ('backend_complete_before_metadata_sha256','backend_complete_round8_before_candidate.json')]:
        req(post.get(field)==base.sha(folder/'audit'/name),'Preserved before-candidate metadata SHA changed')
    req(manifest.get('candidate_subset') is True and manifest.get('factorial_complete') is False
        and manifest.get(source_key.replace('_source_sha256','_factorial_complete')) is False,
        'Candidate subset was mislabeled as full factorial')
    names=[v['name'] for v in manifest['config']['variants']]
    req(manifest.get('executed_candidate_variants')==names,'Executed candidate names differ from producer config')
    return dict(status='PASS',run=str(folder),backend_family=family,candidate_subset=True,factorial_complete=False,
        variants=names,complete_candidate_sha256=base.sha(candidate_path),canonical_sha256=hashes,
        wrapper_sha256=wrapper,source_sha256=sources,config_sha256=candidate['config_sha256'])


def main():
    if '--help' in sys.argv or '-h' in sys.argv: return base.main()
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runs',nargs='+',required=True); parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args(); receipts=[]
    for value in args.runs:
        folder=Path(value)
        if not folder.is_absolute(): folder=base.ROOT/folder if (base.ROOT/folder).is_dir() else base.ROOT/'results'/folder
        for child in ('quality','transfer32'):
            if (folder/child/'records.jsonl').is_file(): folder=folder/child; break
        result=candidate_gate(folder)
        if result and result['status']!='PASS':
            print(json.dumps(result)); return
        if result: receipts.append(result)
    base.main()
    summary=args.output/'summary.json'
    if summary.is_file():
        receipt=dict(status='PASS',scope='Candidate markers verified before delegated label access; inherited scoring/owner/statistics unchanged',
            candidate_runs=receipts,summary_sha256=base.sha(summary),guard_sha256=base.sha(__file__),
            inherited_analyzer_sha256=base.sha(Path(base.__file__)))
        with (args.output/'candidate_completion_audit.json').open('x',encoding='utf-8') as stream:
            json.dump(receipt,stream,indent=2)


if __name__=='__main__': main()
