"""Pinned first-question diagnostics; never counted as generated answers."""
import hashlib,json,sys
from pathlib import Path
import run_continuous

def main():
 config=Path(sys.argv[sys.argv.index('--config')+1]);output=Path(sys.argv[sys.argv.index('--output')+1]);c=json.loads(config.read_text())
 assert c['usage']=='canonical_reference_diagnostics_only'
 fields=('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path','affine_factor_path','decoder_book_path')
 inputs={v[key]:hashlib.sha256(Path(v[key]).read_bytes()).hexdigest() for v in c['variants'] for key in fields if v.get(key)}
 assert inputs==c['external_codec_inputs_sha256']
 run_continuous.main()
 manifest=json.loads((output/'manifest.json').read_text());manifest.update(external_codec_inputs=inputs,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
  frozen_contract_sha256=c['frozen_contract_sha256'],task_accuracy_eligible=False,reference_use='teacher-context diagnostics only; no free answers in this job')
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 receipt=json.loads((output/'complete.json').read_text());receipt.pop('paid_calls',None);receipt.update(full_model_gpu_inference=True,task_accuracy_eligible=False)
 (output/'complete.json').write_text(json.dumps(receipt))

if __name__=='__main__':main()
