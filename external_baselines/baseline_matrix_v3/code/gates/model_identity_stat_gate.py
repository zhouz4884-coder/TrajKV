"""CPU check of cached identity reuse on three real, official model trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import evaluate
from model_registry import get_model_spec


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(os.environ['BASELINE_DATA_ROOT'])
    binding=json.loads((root/'audit/model_stat_binding_v1.json').read_text())
    records=[]
    for model in binding['models']:
        started=time.perf_counter();spec=get_model_spec(model['model_id'])
        result=evaluate.model_identity({'model_path':model['model_path']},spec)
        checks=result['verification_audit']['checks']
        weights=[x for x in checks if x['file'].endswith('.safetensors')]
        assert weights and all(not x['content_hash_recomputed_now'] for x in weights)
        metadata=[x for x in checks if not x['file'].endswith('.safetensors')]
        assert all(x['content_hash_recomputed_now'] for x in metadata)
        records.append({'model_id':model['model_id'],'status':'PASS','elapsed_ms':1000*(time.perf_counter()-started),
                        'weight_files_reusing_prior_full_hash':len(weights),'metadata_files_freshly_hashed':len(metadata),
                        'identity':result})
    report={'status':'PASS_CACHED_IDENTITY_THREE_REAL_MODELS_CPU_ONLY','records':records,
            'CUDA_inference_performed':False,'weights_rehashed_now':False,
            'limitation':'Stat reuse is an integrity fast path to prior official full hashes, not a new full hash pass.',
            'source_sha256':hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({'status':report['status'],'records':[{k:x[k] for k in ('model_id','elapsed_ms','weight_files_reusing_prior_full_hash')} for x in records]}))


if __name__=='__main__':main()
