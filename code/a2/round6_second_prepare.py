"""Create independent LLaVA32x32 identity factors/static ranks, without model use."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke-config',type=Path,required=True)
    p.add_argument('--book-local',type=Path,required=True)
    p.add_argument('--book-remote',required=True)
    p.add_argument('--remote-project',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=8)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.smoke_config.read_text())
    book_hash=hashlib.sha256(args.book_local.read_bytes()).hexdigest()
    nl,nh=32,32
    factor=dict(schema='trajkv.round6.second-identity-metadata.v1',geometry={'layers':nl,'kv_heads':nh,'head_dim':128},
        calibration_mode='identity_no_task_data',cases=0,images=0,labels_used=False,model_weights_frozen=True,
        encoder_book_sha256=book_hash,value_group_size='all',key_gain_mode='none',parameter_count=3*nl*nh,
        key_head_factors=[[1.]*nh for _ in range(nl)],value_head_factors=[[1.]*nh for _ in range(nl)],
        value_center_offsets=[[0.]*nh for _ in range(nl)],
        scope='new L32/Hkv32 identity encoder metadata; no InternVL calibrated arrays imported')
    factor_path=args.output/'identity_factors_l32_h32.json'
    factor_path.write_text(json.dumps(factor,indent=2))
    remote_folder=args.remote_project.rstrip('/')+'/configs/a2/round6_v1/'+args.output.name
    remote_factor=remote_folder+'/'+factor_path.name
    # All 2048 priorities are new. Both static controls use the same K,V axis
    # alternation; only the source-independent order inside each axis differs.
    spread=[int(f'{i:010b}'[::-1],2) for i in range(nl*nh)]
    randomized=list(range(nl*nh));random.Random(1212).shuffle(randomized)
    orders={}
    for name,order in [('spread',spread),('random1212',randomized)]:
        orders[name]=[[axis,index//nh,index%nh] for index in order for axis in ('k','v')]
        assert len({tuple(x) for x in orders[name]})==2*nl*nh
    assert [x[0] for x in orders['spread']]==[x[0] for x in orders['random1212']]
    (args.output/'static_rank_l32_h32.json').write_text(json.dumps(dict(schema='trajkv.round6.second-static-ranks.v1',
        geometry={'layers':nl,'kv_heads':nh,'head_dim':128},axis_sequence='alternating k,v',
        seed=1212,calibration_data_used=False,orders=orders),indent=2))
    full=dict(name='r6_llava15_full',method='full',cap=8,selector='none',store_selector_meta=False)
    variants=[full]
    for order_name,order in orders.items():
        for policy,label in [('incumbent','width'),('v_pair','pair')]:
            variants.append(dict(name='r6_llava15_'+order_name+'_'+label,method='round6_depth',axis_control=order_name,
                cap=8,selector='none',store_selector_meta=False,direction_book_path=args.book_remote,
                metadata_factor_path=remote_factor,v_group_size='all',selected_units=order,
                budget_bits=1.8,depth_policy=policy))
    config=dict(schema='trajkv.round6.second-packed-config.v1',arguments=source['arguments'],start=0,count=args.count,
        expected_visual_tokens=576,variants=variants)
    (args.output/'packed8_all_static.json').write_text(json.dumps(config,indent=2))
    minimum=copy.deepcopy(config);minimum['variants']=variants[:3]
    (args.output/'packed8_minimum.json').write_text(json.dumps(minimum,indent=2))
    print(json.dumps({'output':str(args.output),'factor_bytes':factor['parameter_count']*4,
        'ranking_bytes_each':2*len(orders['spread']),'minimal_variants':len(minimum['variants']),
        'all_variants':len(variants),'book_sha256':book_hash}))


if __name__=='__main__':
    main()
