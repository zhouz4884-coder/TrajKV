"""Diagonal query metric; calibration on old images only, no label fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='query_calibration20_gpu1_v1');a=p.parse_args();folder=ROOT/'results'/a.run
 assert json.loads((folder/'complete.json').read_text())['status']=='COMPLETE'
 sums=np.zeros((28,4,128));count=np.zeros(28);ids=[];inputs={}
 for f in folder.rglob('records.jsonl'):
  inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  for line in f.read_text().splitlines():
   r=json.loads(line);ids.append(r['session_id'])
   assert not r['task_accuracy_eligible']
   for z in r['query_second_moments']:sums[z['layer']]+=np.array(z['sum_q2']);count[z['layer']]+=z['count_per_kv_head']
 assert len(ids)==20 and len(set(ids))==20 and (count>0).all()
 rms=np.sqrt(sums/count[:,None,None]);normalized=rms/np.maximum(np.median(rms,axis=-1,keepdims=True),1e-12)
 # Positive diagonal metric. Clipping is fixed before test and stabilizes inverse
 # query scaling; no parameters are chosen using held-out outcomes.
 scales=np.clip(normalized,.1,10)
 out={'schema':'trajkv.query-diagonal.v1','shape':[28,4,128],'rms_scales':scales.tolist(),'count_per_layer':count.tolist(),'ids':ids,
  'rule':'sqrt(mean(q^2)) / per-head channel median, clipped [0.1,10]; test exponents 0.5 and1, no labels',
  'calibration_inputs':inputs,'dtype_at_encoding':'FP16','role':'shared calibrated transform constants, must be charged in runtime owner'}
 dest=ROOT/'configs/a2/query_scales20.json'
 if dest.exists():raise RuntimeError('Do not overwrite calibrated scales')
 dest.write_text(json.dumps(out,indent=2));print(json.dumps({'images':20,'shape':out['shape'],'output':str(dest)}))
if __name__=='__main__':main()
