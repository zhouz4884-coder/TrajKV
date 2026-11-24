"""Prepare unsealed v5: final reviewed cross-line scanner fix only.

No GPU, scoring, prior scope/source mutation, sealing, export or publication.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BASE=Path('${EVIDENCE_ROOT}/results/round12')
ROOT=Path('${PROJECT_ROOT}')
LOCKS={
 'ops/round12_public_security_v1.py':'c237d09b3475eb49c9f92dcf9da7cb1e34b862c686be23aae6525474857106e9',
 'ops/check_round12_security_repair_v1.py':'9dfc6aa01d223f3aa4d6c0392c4cf58c56162032600f86c0069e2466f8c60240',
 'ops/ROUND12_SECURITY_REPAIR_v1.md':'6c5507846e8c56bbb02b8565c5d59778bf5fa6507645ea73468cd7a8880ad45d'}
FROZEN={'ops/export_public.py':'ae711282f004b6704d76e2f09901ef840480d505706b50bf44a3fd8a02a030ef',
 'ops/export_public_round9_formal_v6.py':'70a28113325e686240970956f610296a809a5157ec67ae0b8e62a50e0476a1cd'}


def require(v,message):
    if not v:raise ValueError('R12_V5_SCOPE: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,obj):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    before_path=BASE/'final_scope_v4/R12_SCOPE_UNSEALED.json'
    original=read(before_path);science=read(BASE/'final_scope_v3/R12_SCOPE_UNSEALED.json')
    require(original['stage_state']=='READY_FOR_ROOT_SEAL' and original['root_seal'] is None,'expected unchanged unsealed v4')
    scope=deepcopy(original);indexed={x['public_path']:x for x in scope['files']};changes=[]
    for public,digest in LOCKS.items():
        item=indexed[public];require(sha(item['source_path'])==digest,'actual reviewed new source differs: '+public)
        previous=deepcopy(item);item['sha256']=digest
        changes.append({'before':previous,'after':deepcopy(item)})
    for public,digest in FROZEN.items():
        require(indexed[public]['sha256']==digest and sha(indexed[public]['source_path'])==digest,'frozen source changed')
    pin='evidence/delivery/CACHED_EXPORT_FINGERPRINTS_v1.json'
    require(indexed[pin]['sha256']=='1d6032dff97de7c282eb4ed967146de4754ea9e17852c1201e1e24f9eb7e8c2f'
            and sha(indexed[pin]['source_path'])==indexed[pin]['sha256'],'Root cache fingerprints changed')
    receipt=BASE/'security_repair_v1/checks_v3/CHECKS.json';checks=read(receipt)
    require(sha(receipt)=='23cde518069375050ea7e885c44135905cfd4a8bd85c3cde49d33c418481c581'
            and checks['status']=='PASS_R12_EXACT_SCANNER_AND_PINNED_REUSE_CHECKS'
            and len(checks['checks'])==11
            and 'non_JSONL_cross_line_whitespace_pattern_rejected_with_original_wholebody_semantics' in checks['checks'],
            'actual final11-check receipt differs')
    keys=('cells','views','comparisons','qa0','attempts','groups','expected_accounting','report_summary')
    require(all(scope.get(k)==original.get(k)==science.get(k) for k in keys),'science changed from v3/v4')
    require(len(scope['cells'])==90 and scope['expected_accounting']['new_successful_terminal_answers']==2200
            and scope['expected_accounting']['pilot_successful_terminal_answers']==2,'cutoff accounting differs')
    require(all(indexed[x['public_path']]==x for x in original['files'] if x['public_path'] not in LOCKS),
            'another original file entry changed')
    output=BASE/'final_scope_v5';require(not output.exists(),'preserve existing scopes; output must be new')
    output.mkdir(parents=True,exist_ok=False)
    def add(path,public,role):
        require(public not in indexed,'new public member collision')
        item={'source_path':str(path),'public_path':public,'sha256':sha(path),'role':role,'delivery_channel':'BROWSER_AND_ARCHIVE'}
        indexed[public]=item;scope['files'].append(item)
    add(receipt,'evidence/technical/R12_EXACT_SCANNER_AND_PINNED_REUSE_CHECKS_v3.json','technical')
    add(Path(__file__).resolve(),'ops/prepare_round12_scope_v5_crossline_revision_v1.py','code')
    manifest={'schema':'trajkv.round12.crossline_security_scope_revision.v1','status':'PREPARED_V5_NOT_SEALED',
        'previous_scope_sha256':sha(before_path),'original_v3_science_scope_sha256':sha(BASE/'final_scope_v3/R12_SCOPE_UNSEALED.json'),
        'changed_original_file_entries':changes,'actual_checks':11,'checks_receipt_sha256':sha(receipt),
        'non_JSONL_scan':'whole body retains original cross-line whitespace regex semantics',
        'JSONL_scan':'per legal JSON row; complete JSON validity is verified by the portable core',
        'exact_scanner_self_match_review':'same original full source, relative path and four exact line hashes; no other exemption',
        'scientific_fields_unchanged_from_v3_and_v4':list(keys),'frozen_original_sources':FROZEN,
        'Root_cache_pin_preserved':True,'prior_scopes_modified':False,'new_GPU_calls':0,'new_task_answers':0,
        'new_scoring_calls':0,'stage_sealed':False,'export_performed':False,'GitHub_mutation':False}
    manifest_path=output/'CROSSLINE_SECURITY_SCOPE_REVISION_v1.json';write(manifest_path,manifest)
    add(manifest_path,'evidence/delivery/CROSSLINE_SECURITY_SCOPE_REVISION_v1.json','protocol')
    browser=[x for x in scope['files'] if x.get('delivery_channel')=='BROWSER_AND_ARCHIVE'
             or (x.get('delivery_channel') is None and x.get('expected_in_git',True))]
    sizes=[{'public_path':x['public_path'],'source_bytes':Path(x['source_path']).stat().st_size} for x in browser]
    require(not [x for x in sizes if x['source_bytes']>16*1024**2],'over-limit browser source')
    audit={'schema':'trajkv.round12.v5_browser_size_audit.v1','status':'PASS_ALL_BROWSER_SOURCE_SIZES',
           'browser_limit_bytes':16*1024**2,'over_limit_count':0,
           'browser_sources_before_this_audit':len(browser),'largest_browser_source':max(sizes,key=lambda x:x['source_bytes']),
           'original_science_raw_not_scanned_or_recomputed':True}
    audit_path=output/'BROWSER_SOURCE_SIZE_AUDIT_v3.json';write(audit_path,audit)
    add(audit_path,'evidence/delivery/BROWSER_SOURCE_SIZE_AUDIT_v3.json','technical')
    scope['stage_state']='READY_FOR_ROOT_SEAL';scope['root_seal']=None
    scope['crossline_security_revision']={'source_scope_sha256':sha(before_path),
        'reviewed_helper_sha256':LOCKS['ops/round12_public_security_v1.py'],
        'checks_receipt_sha256':sha(receipt),'scientific_fields_unchanged_from_v3_and_v4':True,'stage_sealed':False}
    target=output/'R12_SCOPE_UNSEALED.json';write(target,scope)
    print(json.dumps({'status':'PREPARED_SCOPE_V5_NOT_SEALED','scope':str(target),'sha256':sha(target),
        'cells':len(scope['cells']),'files':len(scope['files']),'counts':scope['expected_accounting'],
        'science_identical_to_v3_and_v4':True,'original_modified_tool_entries':3,
        'browser_sources_over_limit':0,'new_GPU_calls':0,'stage_sealed':False,'export_performed':False,'GitHub_mutation':False}))


if __name__=='__main__':main()
