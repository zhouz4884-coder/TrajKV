"""Frozen-method first-answer proxies with all external codec inputs pinned."""
import hashlib,json,sys
from pathlib import Path
import run_continuous
def main():
 config=Path(sys.argv[sys.argv.index('--config')+1]);output=Path(sys.argv[sys.argv.index('--output')+1]);c=json.loads(config.read_text())
 inputs={v[k]:hashlib.sha256(Path(v[k]).read_bytes()).hexdigest() for v in c['variants'] for k in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path') if v.get(k)}
 run_continuous.main()
 (output/'external_codec_inputs.json').write_text(json.dumps({'inputs':inputs,'frozen_contract_sha256':c['frozen_contract_sha256'],'scope':'first-question canonical answer proxies; never task accuracy'},indent=2))
if __name__=='__main__':main()
