"""Timing workload fixed from the frozen candidates, not their test ranking."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 path=ROOT/'audit/a2/frozen_confirmation_v2_mechanism.json';f=json.loads(path.read_text());variants={v['name']:v for v in f['variants']}
 names=['full8','base2_8','k2_lloyd_v1','storage_primary','direction_vq2_v1','lloyd_rms125_joint','read_none','read_all','read_periodic','read_prefill_only','read_risk']
 base=json.loads((ROOT/'configs/a2/mechanism_confirm64_s0_v1.json').read_text());base['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/dev8_short.json',turns=6)
 cfg={'arguments':base['arguments'],'images':3,'repeats':4,'tokens_per_turn':8,'variants':[variants[n] for n in names],
  'frozen_contract_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'declaration':'timing configurations chosen before new64 outcome inspection'}
 name='frozen_timing_gpu0_v1';(ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_timing_multiturn.py','--config',REMOTE_ROOT+f'/configs/a2/{name}.json','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
 print(json.dumps({'measurement_trials':3*4*len(names),'warmups':3*len(names),'launch':'only after both external128 GPU jobs complete and both GPUs are quiet'}))
if __name__=='__main__':main()
