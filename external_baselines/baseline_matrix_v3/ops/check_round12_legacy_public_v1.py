"""CPU admission checks against exact historical actual sources; no GPU/network."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from round12_public_core_v1 import LocalSource, physical_row, require, sha
from round12_public_legacy_v1 import binding


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    scope=json.loads(a.scope.read_text(encoding='utf-8-sig'))
    source=LocalSource(scope,(Path(__file__).resolve().parents[1],Path('${EVIDENCE_ROOT}')))
    tests=[]
    selected=[]
    for label in ('FIXED500_FP16','FIXED500_KIVI_source_example_g32R32','DEV_FP16','FIXED500_R7low'):
        c=next(c for c in scope['cells'] if c['label']==label)
        m,d=(source.doc(c['paths'][k]) for k in ('manifest','complete'))
        binding(c,source,m,d)
        tests.append({'test':'actual_original_producer_'+label,'status':'PASS'})
        selected.append(c)
    def rejected(name,call):
        try:
            call()
        except ValueError as exc:
            tests.append({'test':name,'status':'PASS_EXPECTED_REJECTION','reason':str(exc)})
        else:
            raise AssertionError('negative admission was not rejected: '+name)
    c=deepcopy(selected[2]);c['kind']='new_round12'
    rejected('historical_source_cannot_be_new_inference',lambda:binding(c,source,source.doc(c['paths']['manifest']),source.doc(c['paths']['complete'])))
    c=deepcopy(selected[3]);first=next(iter(c['producer']['files']));del c['producer']['files'][first]
    rejected('original_proof_member_omitted',lambda:binding(c,source,source.doc(c['paths']['manifest']),source.doc(c['paths']['complete'])))
    c=selected[0]
    class WrongLine:
        def __getattr__(self,key):return getattr(source,key)
        def lines(self,path):
            rows=source.lines(path)
            if path==c['producer']['provenance']:
                rows=deepcopy(rows);rows[0][0]['source_exact_line_sha256']='0'*64
            return rows
    rejected('derived_row_wrong_original_exact_line',lambda:binding(c,WrongLine(),source.doc(c['paths']['manifest']),source.doc(c['paths']['complete'])))
    row=deepcopy(source.lines(c['paths']['records'])[0][0]);physical_row(row,reused=True)
    tests.append({'test':'actual_native_disjoint_regions_capacity','status':'PASS'})
    bad=deepcopy(row);bad['owner_before_suffix']['stored_visual_KV_bytes']-=1
    rejected('native_visual_region_wrong_capacity',lambda:physical_row(bad,reused=True))
    bad=deepcopy(row);bad['original_full_prefix_released']=True
    rejected('native_region_requires_explicit_native_owner',lambda:physical_row(bad,reused=True))
    out={'status':'PASS_ACTUAL_HISTORICAL_PUBLIC_ADMISSION_CHECKS','tests':tests,
         'input_scope_sha256':sha(a.scope),'new_model_calls':0,'new_GPU_calls':0,
         'credentials_accessed':False,'network_calls':0,'GitHub_mutation':False,
         'historical_inference_not_counted_as_new':True,'actual_stage_sealed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':out['status'],'tests':len(tests),'output':str(a.output)}))


if __name__=='__main__':
    main()
