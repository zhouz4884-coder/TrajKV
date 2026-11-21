"""Round9 same-size K reassignment and honest global-prefix resource views."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capacity(owners):
    # The public owner audit is still authoritative; this is a local summary.
    storages={}
    for owner in owners:
        for t in owner.named_tensors().values():
            if t.numel():storages[(str(t.device),t.untyped_storage().data_ptr())]=t.untyped_storage().nbytes()
    return sum(storages.values())


@torch.no_grad()
def build_round9_extended_owners(native_prefix_cache,visual_positions,variant,*,candidate,**kwargs):
    from r7_legacy_bridge_v2 import build_r7_owners
    mode=candidate['mode']
    assert mode in ('K-MSE','K-Func','COMBO','TIER')
    assert variant['name']=='r7_joint_k1_gt_cond'
    value_mode=candidate.get('value_mode','original')
    if mode=='COMBO' and value_mode!='original':
        from round9_value_bank_v1 import build_round9_value_owners
        owners,metadata=build_round9_value_owners(native_prefix_cache,visual_positions,variant,
            mode=value_mode,theta_path=candidate.get('theta_path'),theta_sha256=candidate.get('theta_sha256'),**kwargs)
    else:owners,metadata=build_r7_owners(native_prefix_cache,visual_positions,variant,**kwargs)
    before=_capacity(owners)
    metadata=dict(metadata)
    if mode in ('K-MSE','K-Func','COMBO'):
        assert _sha(candidate['book_path'])==candidate['book_sha256']
        raw=json.loads(Path(candidate['book_path']).read_text(encoding='utf-8-sig'))
        assert raw.get('original_book_sha256')==kwargs['expected_asset_sha256']['direction_book_path']
        book=torch.tensor(raw['centers'],dtype=torch.float16,device=visual_positions.device)
        assert tuple(book.shape)==(16,4) and torch.isfinite(book).all()
        from fast_wht import wht
        from round7_key_codec import encode_key,decode_key,encode_sign_depth
        shared=owners[0].shared
        assert all(o.shared is shared for o in owners)
        gain_factors=shared['encoder_factor.key_head_factors']
        changes=[]
        for index,(owner,layer) in enumerate(zip(owners,native_prefix_cache.layers)):
            state=owner.state
            native=layer.keys.index_select(2,visual_positions)
            rotated=wht(native.float()-state.direction_mean.float())
            code,gain=encode_key(rotated,book,amplitude='norm',assignment='l2',
                gain_factor=gain_factors[index][None,:,None,None],iterations=int(variant.get('key_bcd_iterations',1)),
                chunk=int(variant.get('key_bcd_chunk',512)))
            changes.append({'layer':index,'changed_code_bytes':int((code!=state.direction_code).sum()),
                            'changed_gain_values':int((gain!=state.direction_gain).sum())})
            assert code.shape==state.direction_code.shape and code.dtype==state.direction_code.dtype
            state.direction_code,state.direction_gain=code,gain
            base=decode_key(code,gain,book)
            for head in range(4):
                slots=[int(s) for s in state.depth_residual_lookup[0,head].tolist() if int(s)>=0]
                if not slots:continue
                rc,ra,_,_=encode_sign_depth(rotated[0,head],base[0,head],len(slots))
                for stage,slot in enumerate(slots):
                    state.depth_residual_code[slot].copy_(rc[stage])
                    state.depth_residual_alpha[slot].copy_(ra[stage])
        shared['book']=book
        metadata.update(round9_mode=mode,candidate_book_sha256=candidate['book_sha256'],
            K_assignment_and_residual_reencoded=True,K_changes=changes,
            V_candidate=value_mode,original_profile_numeric_fields_unchanged=False)
        assert _capacity(owners)==before,'Same-capacity book/payload must not add bytes'
    else:
        count=candidate['tier_count'];storage=candidate['tier_axis']=='storage'
        assert count in (0,21,43) and candidate['tier_axis'] in ('storage','read')
        original=metadata['selected_stages'];assert len(original)==43
        retained=original[:count]
        shared=owners[0].shared
        lookup=torch.full_like(shared['global_residual_lookup'],-1)
        for index,owner in enumerate(owners):
            state=owner.state
            rows=[(axis,head,stage,int(state.depth_residual_lookup[0 if axis=='k' else 1,head,stage]))
                  for axis,layer,head,stage in retained if layer==index]
            if storage:
                n=visual_positions.numel()
                codes=torch.empty((len(rows),n,4),dtype=torch.int32,device=visual_positions.device)
                alphas=torch.empty((len(rows),n),dtype=torch.float16,device=visual_positions.device)
                for new_slot,(axis,head,stage,old_slot) in enumerate(rows):
                    assert old_slot>=0
                    codes[new_slot].copy_(state.depth_residual_code[old_slot])
                    alphas[new_slot].copy_(state.depth_residual_alpha[old_slot])
                    lookup[index,0 if axis=='k' else 1,head,stage]=new_slot
                # Owning independent allocations, never a view of the old pool.
                state.depth_residual_code,state.depth_residual_alpha=codes,alphas
            else:
                for axis,head,stage,old_slot in rows:
                    lookup[index,0 if axis=='k' else 1,head,stage]=old_slot
            state.depth_residual_lookup=lookup[index]
        shared['global_residual_lookup']=lookup
        metadata.update(round9_mode='TIER',tier_axis=candidate['tier_axis'],tier_count=count,
            selected_stage_count=count,selected_stages=retained,original_selected_stages=original,
            original_storage_residual_units=43 if not storage else count,
            storage_repacked=storage,slice_view_claimed_as_storage_saving=False)
        after=_capacity(owners)
        expected_saved=(43-count)*18*visual_positions.numel() if storage else 0
        assert before-after==expected_saved,('Physical pool capacity did not match',before,after,expected_saved)
    metadata['returned_owners_unique_tensor_capacity_bytes']=_capacity(owners)
    metadata['new_parameter_fit']=mode in ('K-MSE','K-Func','COMBO')
    return owners,metadata
