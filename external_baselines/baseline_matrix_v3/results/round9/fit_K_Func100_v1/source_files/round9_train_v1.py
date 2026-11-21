"""GPU fitting entrypoint consuming only frozen private calibration shards."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def actual_local_teacher(s):
    import torch
    import torch.nn.functional as F
    keys=torch.cat((s['visual_k'],s['exact_keys']),dim=2)
    values=torch.cat((s['visual_v'],s['exact_values']),dim=2)
    positions=torch.cat((s['visual_positions'],s['exact_positions']))
    order=positions.argsort();positions=positions[order]
    keys=keys.index_select(2,order).repeat_interleave(7,dim=1).contiguous()
    values=values.index_select(2,order).repeat_interleave(7,dim=1).contiguous()
    mask=(positions[None,:]<=s['query_positions'][:,None])[None,None]
    return F.scaled_dot_product_attention(s['q'],keys,values,attn_mask=mask,dropout_p=0.,
                                         is_causal=False,scale=s['scaling'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from round9_calibration_v1 import fit,load_sample_shards,make_legacy_callbacks
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    assert config['purpose']=='calibration_only' and config['kind'] in ('V','K-MSE','K-Func')
    assert not set(config)&{'answers','references','reference_path','development_scores','test_scores'}
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,6)
    torch.manual_seed(9);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    collection=Path(config['collection']);complete=json.loads((collection/'complete.json').read_text())
    assert complete['status']=='PASS_CALIBRATION_32_REAL_MODEL_SOURCES' and complete['expected_sources']==32
    shards=[row['file'] for row in complete['sources']]
    hashes={row['file']:row['sha256'] for row in complete['sources']}
    assert len(shards)==32
    a.output.mkdir(parents=True,exist_ok=False)
    wopath=Path(complete['wo_file']);assert sha(wopath)==complete['wo_sha256']
    wos=torch.load(wopath,map_location='cpu',weights_only=True)
    assert set(wos)==set(range(28))
    samples=load_sample_shards(shards,hashes)
    query_policy=config.get('query_source','compressed_query_same_student_text')
    assert query_policy in ('compressed_query_same_student_text','native_query_same_student_text','native_full_state')
    if query_policy!='compressed_query_same_student_text':
        converted=[]
        for s in samples:
            assert 'native_q' in s and 'native_q_rotated' in s,'Native contrast requires actually captured teacher queries'
            assert torch.equal(s['native_query_positions'],s['query_positions'])
            out=dict(s,q=s['native_q'],q_rotated=s['native_q_rotated'],
                     student_snapshot=dict(s['student_snapshot'],calibration_query_policy=query_policy))
            if query_policy=='native_full_state':
                out.update(exact_keys=s['native_exact_keys'],exact_values=s['native_exact_values'],exact_positions=s['native_exact_positions'])
            converted.append(out)
        samples=converted
    collection_manifest=json.loads((collection/'manifest.json').read_text())
    from r7_legacy_bridge_v1 import _source_pin_paths,_check_loaded_pinned_modules
    assert sha(ROOT/'code/r7_legacy_bridge_v1.py')=='b97241c42cdaea96209ba81644b16483b04784e8202080469635184010caa48b'
    pins=_source_pin_paths(Path(config['legacy_a2']),Path(config['legacy_experiments']),
        Path(config['legacy_scripts']),collection_manifest['config']['kivi_source'],None)
    callbacks=make_legacy_callbacks(config['legacy_a2'],config['legacy_experiments'],config['legacy_scripts'])
    _check_loaded_pinned_modules(pins)
    call_counts={'packed_reader':0,'encode_key':0};progress_started=time.perf_counter()
    def observed(name,original):
        def call(*args,**kwargs):
            value=original(*args,**kwargs)
            call_counts[name]+=1
            if call_counts[name]%256==0:
                print(json.dumps({'phase':'ACTUAL_CALIBRATION_PROGRESS','kind':config['kind'],
                    'callback':name,'calls':call_counts[name],'elapsed_s':time.perf_counter()-progress_started}),flush=True)
            return value
        return call
    for name in call_counts:callbacks[name]=observed(name,callbacks[name])
    assert all(s['purpose']=='calibration' for s in samples)
    sources={str(Path(__file__)):sha(__file__),
             str(ROOT/'code/round9_calibration_v1.py'):sha(ROOT/'code/round9_calibration_v1.py')}
    if config['kind']!='V':sources[str(ROOT/'code/round9_keyfit_v1.py')]=sha(ROOT/'code/round9_keyfit_v1.py')
    write_new(a.output/'manifest.json',{'schema':'trajkv.round9.calibration_fit_run.v1','config':config,
        'config_sha256':sha(a.config),'collection_complete_sha256':sha(collection/'complete.json'),
        'sources':sources,'private_WO_sha256':complete['wo_sha256'],'actual_GPU':torch.cuda.get_device_name(),
        'references_read':False,'calibration_sources':32,'training_events':len(samples),'query_source':query_policy,
        'source_weights_or_private_tensors_public_upload':False})
    device_wos={i:w.to('cuda') for i,w in wos.items()}
    cfg=dict(callbacks,device='cuda:0',wo_by_layer=device_wos,teacher_reader=actual_local_teacher,
        steps=config.get('steps',100),outer_loops=1,learning_rate=config.get('learning_rate',.003),
        seed=9,ls_warmstart=True,output=a.output/'fit',
        collection_provenance={'complete_sha256':sha(collection/'complete.json'),'private_WO_sha256':complete['wo_sha256']},
        legacy_source_hashes={path:row['sha256'] for path,row in pins.items()})
    if config.get('initial_theta_path'):
        assert sha(config['initial_theta_path'])==config['initial_theta_sha256']
        cfg['initial_theta']=config['initial_theta_path']
    begin=time.perf_counter()
    if config['kind']=='V':
        result=fit(samples,cfg);artifact=a.output/'fit/value_theta.json'
    else:
        from round9_keyfit_v1 import fit as fit_key
        objective={'K-MSE':'mse','K-Func':'functional'}[config['kind']]
        result=fit_key(samples,cfg,objective=objective);artifact=a.output/'fit/key_book.json'
    assert artifact.exists()
    assert all(sha(path)==digest for path,digest in sources.items())
    assert all(sha(path)==row['sha256'] for path,row in pins.items())
    torch.cuda.synchronize()
    write_new(a.output/'complete.json',{'status':'PASS_ROUND9_CALIBRATION_FIT','kind':config['kind'],
        'calibration_sources':32,'actual_GPU':True,'artifact':str(artifact),'artifact_sha256':sha(artifact),
        'wall_s':time.perf_counter()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'fit_summary':result.get('fit_summary',result.get('round9_fit',{})),'free_generation_quality_claim':False,
        'student_state_refreshes':0,'frozen_student_snapshot_only':True})
    print(json.dumps({'status':'PASS_ROUND9_CALIBRATION_FIT','kind':config['kind'],
                      'artifact_sha256':sha(artifact),'wall_s':time.perf_counter()-begin}),flush=True)


if __name__=='__main__':main()
