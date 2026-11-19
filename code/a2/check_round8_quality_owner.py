"""Independent CPU accounting fixtures, explicitly not inference records."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

from round8_data_prepare import ROOT,read,sha,new
from round8_budget_check import cost
from round8_quality_owner import audit_owner


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    assert not a.output.exists()
    cfg=read(ROOT/'configs/a2/round7_v1/confirm_doc128_s0.json')
    old=next(v for v in cfg['variants'] if v['name']=='r7_joint_k1_gt_cond')
    oldbook=old['direction_book_path']; factor=old['metadata_factor_path']
    prefix=oldbook.split('/results/',1)[0]
    assets=ROOT/'outputs/round8_quality_cpu_controls_v1'
    books={'k8':assets/'k8_product.json','kpos':assets/'k4_position.json','v8':assets/'v8_cube.json'}
    from round8_quality_owner import local_path
    factors=read(local_path(factor)); lh=factors['parameter_count']//3; layers=len(factors['key_head_factors']); heads=lh//layers
    controls=[]; fixtures=[]
    full_priority=old['selected_stages']; assert len(full_priority)==224
    stages=full_priority[:43]
    specs=[('00',4,16,1,False,False),('10',8,256,1,False,False),('01',4,16,1,True,False),
           ('11',8,256,1,True,False),('Kpos',4,16,32,False,False),('Kdecoder',8,256,1,False,True)]
    for name,kd,kc,pos,hasv,decoder in specs:
        for ranking_count in (43,224):
            requested=copy.deepcopy(stages)
            if ranking_count>43:
                requested=copy.deepcopy(full_priority)
                assert len(requested)==224
            kb=oldbook if kc==16 and pos==1 else str(books['kpos' if pos>1 else 'k8'])
            variant=dict(name='CPU_'+name+str(ranking_count),method='round8_joint',direction_book_path=kb,
                metadata_factor_path=factor,metadata_reference_book_path=oldbook,
                key_encoding='decoder-only' if decoder else 'reencode',value_encoding='cube-identity' if hasv else 'binary',
                selected_stages=requested,res_stages_limit=43,budget_bits=1.8)
            if decoder: variant['encoder_book_path']=oldbook
            if hasv: variant['value_joint_book_path']=str(books['v8'])
            external={path:sha(local_path(path)) for key,path in variant.items() if key.endswith('_path')}
            budget=cost(1792,layers,heads,kd,kc,8 if hasv else None,256 if hasv else None,
                        kpos=pos,decoder_only=decoder,ranking_entries=ranking_count)
            fields=budget['actual_owner_components']; full=budget['visual_FP16_denominator_bytes']
            base=sum(fields[k] for k in ('K_codes','K_row_gain','K_mean','V_codes','V_channel_scale_min','K_global_or_position_books','V_book'))
            encoder=sum(fields[k] for k in ('K_retained_old_encoder_book','complete_uint16_ranking','policy_descriptor','encoder_factors'))
            ranking=struct.pack('<'+'H'*len(requested),*[((int(axis=='v')*layers+l)*heads+h)*2+s for axis,l,h,s in requested])
            policy=struct.pack('<Bf',3,0.)
            record=dict(method='round8_joint',variant=variant['name'],variant_config=variant,session_id='CPU_ONLY',
                visual_fp16_materialization=False,key_source_fp16_retained=False,depth_residual_source_fp16_retained=False,
                question_state_scope='recomputed_under_codec',initial_logits_shared_fp16=False,first_question_suffix_replay_tokens=1,
                metadata_reference_book_sha256=external[oldbook],metadata_calibration_sha256=external[factor],
                key_book_shape=list(np.asarray(read(local_path(kb))['centers']).shape),key_book_words=kc,direction_book_dimension=kd,
                key_position_specific=pos>1,key_index_bytes_per_token=16,decoder_book_bytes=pos*kc*kd*2,
                key_book_sha256=external[kb],key_encoding=variant['key_encoding'],key_encoder_extra_book_bytes=128 if decoder else 0,
                key_encoder_book_sha256=external[oldbook] if decoder else None,value_encoding=variant['value_encoding'],
                value_frame='all',value_row_gain_added=False,value_joint_book_bytes=4096 if hasv else 0,
                value_joint_book_sha256=external[str(books['v8'])] if hasv else None,value_index_bytes_per_token=16,visual_tokens=1792,
                depth_residual_selected_stages=stages,depth_residual_requested_stages=requested,
                encoder_calibration_parameters_bytes=encoder,depth_residual_complete_bytes=budget['complete_visual_owner_bytes'],
                depth_residual_encoder_ranking_bytes=len(ranking),depth_residual_encoder_policy_bytes=5,
                depth_residual_encoder_ranking_sha256=hashlib.sha256(ranking).hexdigest(),
                depth_residual_encoder_policy_sha256=hashlib.sha256(policy).hexdigest(),
                depth_residual_actual_budget_bits=budget['complete_bits_per_KV_scalar'])
            record['owned_initial']=dict(full_visual_retained=False,visual_fp16_denominator=full,visual_base=base,
                visual_residual=fields['selected_sign_alpha_pool'],selector_metadata=fields['full_stage_lookup'],
                visual_position=fields['shared_visual_positions'],encoder_constant_bytes=encoder,
                complete_visual_owner_bytes=budget['complete_visual_owner_bytes'],total_unique=10**12)
            actual=audit_owner(record,variant,external,{})
            assert actual['complete_visual_owner_bytes']==budget['complete_visual_owner_bytes']
            fixtures.append((record,variant,external)); controls.append(dict(control=name,requested_ranking=ranking_count,bytes=actual['complete_visual_owner_bytes'],status='PASS'))
    bad,variant,external=copy.deepcopy(fixtures[-1]); bad['owned_initial']['encoder_constant_bytes']+=128
    try: audit_owner(bad,variant,external,{})
    except ValueError: rejected=True
    else: rejected=False
    assert rejected
    out=dict(status='PASS',scope='CPU metadata/accounting fixtures only; no R8 model execution or accuracy evidence',
        controls=controls,double_encoder128_rejected=rejected,total_unique_ignored=True,GPU_used=False,new_model_forward_count=0,
        helper_sha256=sha(Path(__file__).with_name('round8_quality_owner.py')),check_sha256=sha(__file__),
        independent_arithmetic_sha256=sha(Path(__file__).with_name('round8_budget_check.py')))
    new(a.output,out); print(json.dumps(dict(status='PASS',controls=len(controls),output=str(a.output),GPU_used=False)))


if __name__=='__main__': main()
