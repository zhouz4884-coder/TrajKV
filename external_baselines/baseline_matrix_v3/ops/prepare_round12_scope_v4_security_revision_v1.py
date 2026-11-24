"""Prepare a new unsealed R12 scope for reviewed publication-tool fixes.

No science recomputation, GPU, source mutation, seal/export or GitHub writes.
The exact frozen sanitizer and R9 streaming-writer bytes remain selected.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

BASE=Path('${EVIDENCE_ROOT}/results/round12')
FROZEN={'ops/export_public.py':'ae711282f004b6704d76e2f09901ef840480d505706b50bf44a3fd8a02a030ef',
        'ops/export_public_round9_formal_v6.py':'70a28113325e686240970956f610296a809a5157ec67ae0b8e62a50e0476a1cd'}
LIMIT=16*1024**2


def require(value,message):
    if not value:raise ValueError('R12_SECURITY_SCOPE_REVISION: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-scope',type=Path,default=BASE/'final_scope_v3/R12_SCOPE_UNSEALED.json')
    p.add_argument('--updates',type=Path,required=True,help='Explicit reviewed actual tool/support file hashes')
    p.add_argument('--output',type=Path,default=BASE/'final_scope_v4')
    a=p.parse_args()
    require(a.source_scope.resolve().is_relative_to(BASE.resolve()),'source scope outside stage evidence root')
    require(a.output.resolve().is_relative_to(BASE.resolve()) and not a.output.exists(),'new E data-disk output required')
    spec=read(a.updates)
    require(spec.get('schema')=='trajkv.round12.public_tool_updates.v1' and isinstance(spec.get('files'),list)
            and spec['files'],'explicit reviewed tool update schema required')
    original=read(a.source_scope)
    require(original['stage_state']=='READY_FOR_ROOT_SEAL' and original.get('research_cutoff_complete') is True,
            'expected v3 completed unsealed source scope')
    require(len(original['cells'])==90 and original['expected_accounting']['new_successful_terminal_answers']==2200
            and original['expected_accounting']['pilot_successful_terminal_answers']==2,'scientific cutoff differs')
    scope=deepcopy(original)
    indexed={x['public_path']:x for x in scope['files']}
    changes=[]
    for item in spec['files']:
        path=Path(item['source_path']).resolve();public=item['public_path']
        rel=PurePosixPath(public)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in public
                and (public.startswith('ops/') or public.startswith('evidence/')),
                'tool revision cannot alter scientific result/config paths')
        require(public not in FROZEN,'frozen sanitizer/streaming writer must not be updated')
        require(path.is_file() and sha(path)==item['sha256'],'reviewed tool body differs: '+public)
        require(item['role'] in ('code','portable_verifier','protocol','support','technical'), 'unsupported new tool role')
        require(path.stat().st_size<=LIMIT,'tool/support body exceeds unchanged browser limit')
        value={'source_path':str(path),'public_path':public,'sha256':item['sha256'],'role':item['role'],
               'delivery_channel':'BROWSER_AND_ARCHIVE'}
        before=deepcopy(indexed.get(public))
        if public in indexed:
            indexed[public].clear();indexed[public].update(value)
        else:
            scope['files'].append(value);indexed[public]=value
        changes.append({'before':before,'after':deepcopy(value)})
    require('ops/round12_public_security_v1.py' in indexed
            and 'ops/export_public_round12_v2.py' in indexed,'new reviewed security helper/exporter required')
    require(any(x['after']['public_path']=='ops/verify_public_round12_v1.py' for x in changes),
            'actual updated cold verifier must be bound explicitly')
    for public,digest in FROZEN.items():
        require(indexed[public]['sha256']==digest and sha(indexed[public]['source_path'])==digest,
                'frozen original source bytes changed')
    full='evidence/QA0/bm_R12_value_offset_diag_gpu1_v1/VALUE_OFFSET_DIAGNOSTIC_v1.json'
    require(indexed[full]['delivery_channel']=='ARCHIVE_ONLY'
            and indexed[full]['sha256']=='d99baa4311131db5988709eb20269c6248bd79ede6560f668c76dac4a36ecc35',
            'previous exact full QA0 archive channel must remain')
    science_keys=('cells','views','comparisons','qa0','attempts','groups','expected_accounting','report_summary')
    require(all(scope.get(k)==original.get(k) for k in science_keys),'science graph changed')
    changed={x['after']['public_path'] for x in changes}
    require(all(indexed[x['public_path']]==x for x in original['files'] if x['public_path'] not in changed),
            'unrelated original file entry changed')
    a.output.mkdir(parents=True,exist_ok=False)
    def add(path,public,role):
        require(public not in indexed,'generated revision member collision')
        value={'source_path':str(path),'public_path':public,'sha256':sha(path),'role':role,
               'delivery_channel':'BROWSER_AND_ARCHIVE'}
        scope['files'].append(value);indexed[public]=value
    add(Path(__file__).resolve(),'ops/prepare_round12_scope_v4_security_revision_v1.py','code')
    add(a.updates.resolve(),'evidence/delivery/ROUND12_REVIEWED_TOOL_UPDATES_v1.json','protocol')
    revision={'schema':'trajkv.round12.public_security_scope_revision.v1',
        'status':'PREPARED_ACTUAL_TOOLS_SCOPE_ONLY_UNSEALED','source_scope_sha256':sha(a.source_scope),
        'updates_spec_sha256':sha(a.updates),'tool_file_changes':changes,
        'frozen_original_dependency_source_sha256':FROZEN,
        'reviewed_exact_byte_public_sources':{'ops/export_public.py':FROZEN['ops/export_public.py']},
        'review_scope':'Only the exact frozen scanner own reviewed regex literals may receive its scoped false-positive mask; no other file or field is exempted',
        'scientific_fields_unchanged':list(science_keys),
        'scientific_fields_canonical_sha256':canonical({k:original.get(k) for k in science_keys}),
        'new_GPU_calls':0,'new_task_answers':0,'new_scoring_calls':0,
        'old_scope_or_failed_stage_modified':False,'browser_limit_raised':False,
        'cache_reuse_must_validate_original_and_public_body_SHA':True,
        'stage_sealed':False,'export_performed':False,'GitHub_mutation':False}
    revision_path=a.output/'PUBLIC_SECURITY_SCOPE_REVISION_v1.json'
    write(revision_path,revision)
    add(revision_path,'evidence/delivery/PUBLIC_SECURITY_SCOPE_REVISION_v1.json','protocol')
    browser=[x for x in scope['files'] if x.get('delivery_channel')=='BROWSER_AND_ARCHIVE'
             or (x.get('delivery_channel') is None and x.get('expected_in_git',True))]
    sizes=[{'public_path':x['public_path'],'source_bytes':Path(x['source_path']).stat().st_size} for x in browser]
    require(not [x for x in sizes if x['source_bytes']>LIMIT],'new scope has over-limit browser source')
    audit={'schema':'trajkv.round12.security_revision_browser_size_audit.v1',
        'status':'PASS_ALL_BROWSER_SOURCE_SIZES','browser_limit_bytes':LIMIT,
        'browser_sources_before_this_audit':len(browser),'over_limit_count':0,
        'largest_browser_source':max(sizes,key=lambda x:x['source_bytes']),
        'new_sources':[x for x in sizes if x['public_path'] not in {v['public_path'] for v in original['files']}],
        'old_large_QA0_still_archive_only':True,'source_scope_sha256':sha(a.source_scope),
        'original_science_raw_not_scanned_or_recomputed':True}
    audit_path=a.output/'BROWSER_SOURCE_SIZE_AUDIT_v2.json';write(audit_path,audit)
    require(audit_path.stat().st_size<=LIMIT,'size audit exceeds browser limit')
    add(audit_path,'evidence/delivery/BROWSER_SOURCE_SIZE_AUDIT_v2.json','technical')
    scope['stage_state']='READY_FOR_ROOT_SEAL';scope['root_seal']=None
    scope['public_security_revision']={'schema':revision['schema'],
        'source_scope_sha256':revision['source_scope_sha256'],'updates_spec_sha256':revision['updates_spec_sha256'],
        'scientific_fields_canonical_sha256':revision['scientific_fields_canonical_sha256'],
        'reviewed_exact_scanner_original_sha256':FROZEN['ops/export_public.py'],
        'new_stage_required':True,'stage_sealed':False}
    target=a.output/'R12_SCOPE_UNSEALED.json';write(target,scope)
    require(sha(a.source_scope)==revision['source_scope_sha256'],'original scope changed during preparation')
    print(json.dumps({'status':'PREPARED_SCOPE_V4_NOT_SEALED','scope':str(target),
        'cells':len(scope['cells']),'files':len(scope['files']),'updated_or_added_tool_files':len(changes),
        'new_GPU_calls':0,'new_task_answers':0,'stage_sealed':False,'export_performed':False,'GitHub_mutation':False}))


if __name__=='__main__':main()
