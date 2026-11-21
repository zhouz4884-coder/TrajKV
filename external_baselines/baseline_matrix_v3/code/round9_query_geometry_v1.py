"""Small calibration-only query geometry probe, never a task-quality result."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
    torch.backends.cuda.matmul.allow_tf32=False
    done=json.loads((a.collection/'complete.json').read_text());assert done['status']=='PASS_CALIBRATION_32_REAL_MODEL_SOURCES'
    assert len(done['sources'])==32
    a.output.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    cov=torch.zeros((28,4,128,128),dtype=torch.float64,device='cuda');counts=torch.zeros((28,4),dtype=torch.int64,device='cuda')
    for item in done['sources']:
        # mmap avoids touching native K/V/W_O payloads; source SHA is anchored
        # by the already closed fitting loaders and producer manifest.
        case=torch.load(item['file'],map_location='cpu',weights_only=True,mmap=True)
        assert case['schema']=='trajkv.round9.calibration_case.v1'
        groups={}
        for e in case['events']:groups.setdefault(e['layer_idx'],[]).append(e)
        assert set(groups)==set(range(28))
        for layer,events in groups.items():
            for e in events:
                q=e['q_rotated'].to(device='cuda',dtype=torch.float64).reshape(4,7,128)
                cov[layer]+=torch.einsum('hqd,hqe->hde',q,q)/(32*len(events)*7)
                counts[layer]+=7
    traces=cov.diagonal(dim1=-2,dim2=-1).sum(-1);assert bool((traces>0).all())
    eig,vec=torch.linalg.eigh(cov);unit=cov/traces[...,None,None]
    global_cov=unit.mean((0,1));geig,gvec=torch.linalg.eigh(global_cov)
    bases=gvec[:,-3:].T.contiguous()
    # Fix sign only for reproducible presentation; uu^T is sign invariant.
    for row in bases:
        if row[row.abs().argmax()]<0:row.mul_(-1)
    projected=torch.einsum('rd,lhde,re->lhr',bases,cov,bases)
    fractions=projected/traces[...,None]
    selected=fractions.argmax(-1);best=fractions.max(-1).values
    rows=[]
    for layer in range(28):
        for head in range(4):
            rows.append({'layer':layer,'head':head,'trace':float(traces[layer,head]),
                'top_eigen_fraction':float(eig[layer,head,-1]/traces[layer,head]),
                'top3_eigen_fraction':float(eig[layer,head,-3:].sum()/traces[layer,head]),
                'shared3_selected_single_fraction':float(best[layer,head]),
                'shared3_full_fraction':float(fractions[layer,head].sum()),'selected_basis':int(selected[layer,head]),
                'q_vectors':int(counts[layer,head])})
    result={'status':'PASS_CALIBRATION_QUERY_GEOMETRY_ONLY','actual_GPU':True,'new_model_forward':0,'new_task_QA':0,
        'calibration_sources':32,'query_source':'actually captured compressed R7 student; no native query supplied',
        'moment':'second moment E[qq^T], not centered covariance','pooling':'equal source, equal phase per source, seven query heads per KV head; global bases average unit-trace heads',
        'collection_complete_sha256':sha(a.collection/'complete.json'),'script_sha256':sha(__file__),
        'global_top3_unit_trace_fraction':float(geig[-3:].sum()),'heads':rows,'shared_bases':bases.float().cpu().tolist(),
        'mean_top_eigen_fraction':sum(r['top_eigen_fraction'] for r in rows)/112,
        'mean_shared_best_fraction':sum(r['shared3_selected_single_fraction'] for r in rows)/112,
        'mean_shared3_full_fraction':sum(r['shared3_full_fraction'] for r in rows)/112,
        'task_quality_or_online_future_prediction_claim':False,'private_PT_WO_or_images_exported':False,
        'wall_s':time.perf_counter()-begin}
    with(a.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    with(a.output/'complete.json').open('x',encoding='utf-8') as f:json.dump({'status':result['status'],'actual_GPU':True,'summary_sha256':sha(a.output/'summary.json'),'new_task_QA':0},f,indent=2)
    print(json.dumps({k:result[k] for k in ['status','global_top3_unit_trace_fraction','mean_top_eigen_fraction','mean_shared_best_fraction','wall_s']}),flush=True)


if __name__=='__main__':main()
