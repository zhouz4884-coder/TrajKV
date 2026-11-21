"""Same-input numerical diagnosis of the failed calibration KL check."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-index',type=int,default=11);a=p.parse_args()
    data=Path(os.environ['BASELINE_DATA_ROOT']);lease=(data/'jobs/queue_gpu0.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import round9_calibration_v1 as cal
    import round9_exchange_calibrate_v1 as ex
    import round9_precision_exchange_v1 as ops
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    complete=json.loads((a.collection/'complete.json').read_text());item=sorted(complete['sources'],key=lambda row:row['source_id'])[a.source_index]
    samples=cal.load_sample_shards([item['file']],{item['file']:item['sha256']})
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for raw in samples:
        if raw['phase']!='suffix_last':continue
        s=cal.to_device(raw,'cuda');f=ex.joint_distribution(s,native=True)
        c=ex.joint_distribution(s,s['r7_fields'],s['book'],decode_key=ops.decode_exchange_key_fields)
        term=f['p']*(f['logp']-c['logp']);r=torch.where(f['visible'][None,None],term,torch.zeros_like(term)).sum(-1)
        row={'source_id':s['source_id'],'id':s['id'],'layer':s['layer_idx'],'dtype':str(f['p'].dtype),
             'visible_all':bool(f['visible'].all()),'result_finite':bool(torch.isfinite(r).all()),
             'result_min':float(r.nan_to_num().min()),'result_max':float(r.nan_to_num().max()),
             'nan_heads':torch.isnan(r).nonzero().cpu().tolist(),
             'p_sum_native_range':[float(f['p'].sum(-1).min()),float(f['p'].sum(-1).max())],
             'p_sum_candidate_range':[float(c['p'].sum(-1).min()),float(c['p'].sum(-1).max())],
             'native_logp_finite':int(torch.isfinite(f['logp']).sum()),'candidate_logp_finite':int(torch.isfinite(c['logp']).sum()),
             'logp_elements':f['logp'].numel(),'q_maxabs':float(s['q'].abs().max()),
             'native_K_maxabs':float(s['visual_k'].abs().max())}
        if not bool(torch.isfinite(r).all()) or float(r.min())< -1e-10:
            torch.save({'native':f,'candidate':c,'result':r,'source':row},a.output/f'bad_layer{s["layer_idx"]}.pt')
            row['bad_distribution_saved']=True
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
    with(a.output/'summary.json').open('x') as f:json.dump({'status':'NUMERIC_DIAGNOSIS_ONLY','rows':rows,'source_shard_sha256':item['sha256'],'new_task_QA':0},f,indent=2)


if __name__=='__main__':main()
