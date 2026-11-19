"""CPU factorial integration controls plus opt-in small actual CUDA reader grid."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from round8_codec import encode_key, decode_key, key_template
from round6_depth_codec import encode_sign_depth, decode_sign_stage, plan_depth_stages
from round8_joint_gain import joint_refit_key
from round8_value_joint_refit import joint_refit_value, cpu_check as value_cpu_check
from round8_check import frozen_reference_functions
from quantizers import binary_centroids
from round8_field_joint_runner import (sha, source_fingerprints, validate_variant,
    validate_config, field_joint_class, scoped_registration, finalize_run)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def rejected(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True
    return False


def priority_fixture():
    root = Path(__file__).resolve().parents[2]
    path = root/'configs/a2/round7_v1/confirm_doc128_s0.json'
    if path.is_file():
        cfg = json.loads(path.read_text(encoding='utf-8-sig'))
        return next(v for v in cfg['variants'] if v['name'] == 'r7_joint_k1_gt_cond')['selected_stages']
    return [[axis, layer, head, 0] for layer in range(28) for head in range(4) for axis in ('k', 'v')]


def physical_bytes(states, book):
    total = book.numel()*book.element_size()
    for state in states:
        total += sum(getattr(state, name).numel()*getattr(state, name).element_size() for name in
            ('direction_code', 'direction_gain', 'v_code', 'v_scale', 'v_min', 'depth_residual_code', 'depth_residual_alpha'))
    return total


def cpu_controls(output):
    torch.set_num_threads(1); torch.manual_seed(20261118)
    value_controls = value_cpu_check()
    decoder, fold = frozen_reference_functions()
    book = torch.randn(16, 4).half()
    bookpath = output/'synthetic_book.json'
    bookpath.write_text(json.dumps(dict(centers=book.float().tolist())), encoding='utf-8')
    priority = priority_fixture()
    common = dict(method='round8_joint', key_assignment='l2', key_amplitude='norm',
        key_encoding='reencode', value_encoding='binary', v_group_size='all',
        direction_book_path=str(bookpath), metadata_reference_book_path=str(bookpath),
        metadata_factor_path='CPU-only-placeholder', round8_factor_policy='inherit_from_reference_book',
        selected_stages=priority, res_stages_limit=43, value_joint_rounds=2, budget_bits=1.8, cap=8)
    variants = [dict(common, name=f'K{int(k)}V{int(v)}', joint_gain_refit=k, value_joint_refit=v)
                for k in (False, True) for v in (False, True)]
    declarations = validate_config(dict(variants=variants))
    require(rejected(lambda: validate_config(dict(variants=variants[:3]))), 'Missing factorial cell accepted')
    require(rejected(lambda: validate_variant(dict(variants[-1], key_assignment='row_cosine_cd'))), 'CD accepted in L2 factorial')
    require(rejected(lambda: validate_variant(dict(variants[-1], value_joint_rounds=3))), 'Nonfixed V rounds accepted')
    require(rejected(lambda: validate_variant(dict(variants[-1], selected_stages=priority[:43]))), 'Truncated rank accepted')
    require(rejected(lambda: validate_variant(dict(variants[-1], value_encoding='reencode'))), 'Joint-book V accepted')
    nv, lh = 1792, 112
    base = lh*nv*34 + lh*128*6 + 128 + nv*8
    plan = plan_depth_stages(layers=28, heads=4, visual_tokens=nv, visual_full_bytes=lh*nv*512,
        base_bytes=base, factor_bytes=1344, budget_bits=1.8, selected_stages=priority, stage_limit=43)
    require(plan['predicted_bytes'] == 8315013 and len(plan['ranking_constant']) == 448, 'Full owner/ranking algebra differs')
    n = 32
    sources_k = torch.randn(28, 1, 4, n, 128)
    sources_v = torch.randn(28, 1, 4, n, 128).half()
    sources_v[:, :, :, :, 0] = .125
    class Base:
        settings = {}
        def _initialize(self, ks, vs, *, fail=False):
            cfg = type(self).settings
            self.states, self.depth_distortions, self.joint_gain_statistics, self.events = [], [], [], []
            self.value_encoding, self.vbook, self.vgroup = 'binary', None, n
            self.joint_gain_enabled = cfg['joint_gain_refit']
            self.requested_residual_stages = [tuple(x) for x in cfg['selected_stages']]
            self.selected_residual_stages = self.requested_residual_stages[:43]
            self.ranking_constant_bytes = 448
            for layer in range(28):
                local = [(axis, head, depth) for axis, layer_id, head, depth in self.selected_residual_stages if layer_id == layer]
                kc, kg = encode_key(ks[layer], book, gain_factor=torch.tensor([.91,.98,1.04,1.11])[None,:,None,None])
                record = binary_centroids(vs[layer].transpose(2,3).contiguous(), group_size=n)
                vc, scale, minimum = fold(record, 1, torch.tensor([.86,.95,1.06,1.14])[None,:,None,None],
                    torch.tensor([-.1,.03,.12,-.06])[None,:,None,None])
                state = SimpleNamespace(direction_code=kc, direction_gain=kg, v_code=vc, v_scale=scale, v_min=minimum,
                    depth_residual_code=torch.empty(len(local), n, 4, dtype=torch.int32),
                    depth_residual_alpha=torch.empty(len(local), n, dtype=torch.float16), fixed_signs={})
                state.original_v_code = vc.clone()
                state.frame_pointers = (scale.data_ptr(), minimum.data_ptr(), state.depth_residual_alpha.data_ptr())
                self.states.append(state)
                if fail and layer == 0:
                    raise ValueError('intentional constructor failure')
                self._encode_axis(layer, state, 'k', local, ks[layer,0], decode_key(kc,kg,book)[0])
                vb = decoder(vc,scale,minimum,n,1).transpose(2,3).contiguous()[0]
                self._encode_axis(layer, state, 'v', local, vs[layer,0], vb)
        def _encode_axis(self, layer, state, axis, selected, source, base):
            self.events.append(('super', axis, layer))
            for head in sorted({head for a, head, _ in selected if a == axis}):
                depth = sum(a == axis and h == head for a,h,_ in selected)
                rc, ra, final, errors = encode_sign_depth(source[head], base[head], depth)
                for stage in range(depth):
                    slot = selected.index((axis,head,stage))
                    state.depth_residual_code[slot].copy_(rc[stage]); state.depth_residual_alpha[slot].copy_(ra[stage])
                    state.fixed_signs[slot] = rc[stage].clone()
                if axis == 'k' and self.joint_gain_enabled:
                    slot = selected.index(('k',head,0))
                    gain, alpha, stats = joint_refit_key(source[head].float(), key_template(state.direction_code[0,head],book),
                        state.direction_gain[0,head], state.depth_residual_code[slot], state.depth_residual_alpha[slot])
                    state.direction_gain[0,head].copy_(gain); state.depth_residual_alpha[slot].copy_(alpha)
                    self.joint_gain_statistics.append(dict(layer=layer,head=head,**stats))
                self.depth_distortions.append(dict(axis=axis, layer=layer, head=head, depth=depth, squared_error_by_depth=errors))
        def result_metadata(self):
            return dict(joint_gain_refit=self.joint_gain_enabled,
                key_gain_rule='joint-effective-gain-and-alpha' if self.joint_gain_enabled else 'norm',
                key_gain_formula='original-super-physical-gain-formula')
    cls = field_joint_class(Base)
    cells, observations = {}, []
    expected_v = sum(x[0] == 'v' for x in priority[:43])
    expected_k = 43-expected_v
    for variant in variants:
        cls.settings = variant
        instance = cls(); instance._initialize(sources_k,sources_v)
        meta = instance.result_metadata()
        k,v = variant['joint_gain_refit'],variant['value_joint_refit']
        require(len(instance.value_joint_statistics) == (expected_v if v else 0), 'Wrong V refit coverage')
        require(len(instance.joint_gain_statistics) == (expected_k if k else 0), 'Wrong K refit coverage')
        require(meta['value_joint_added_retained_tensor_bytes'] == 0 and meta['field_joint_factorial'] == dict(key_joint=k,value_joint=v), 'Metadata/zero field contract failed')
        require(meta['key_gain_formula'] == 'original-super-physical-gain-formula', 'K physical formula overwritten')
        for layer,state in enumerate(instance.states):
            require(torch.equal(state.v_code,state.original_v_code), 'V base bits changed')
            require((state.v_scale.data_ptr(),state.v_min.data_ptr(),state.depth_residual_alpha.data_ptr()) == state.frame_pointers, 'Hook replaced physical slots rather than copying')
            for slot,original in state.fixed_signs.items():
                require(torch.equal(state.depth_residual_code[slot],original), 'Packed residual signs changed')
            if v:
                dense_v = decoder(state.v_code,state.v_scale,state.v_min,n,1).transpose(2,3)[0]
                local = [(axis,head,depth) for axis,layer_id,head,depth in instance.selected_residual_stages if layer_id==layer]
                for axis,head,_ in local:
                    if axis != 'v': continue
                    slot = local.index(('v',head,0))
                    final = decode_sign_stage(dense_v[head],state.depth_residual_code[slot],state.depth_residual_alpha[slot])
                    error = float((sources_v[layer,0,head].double()-final.double()).square().sum())
                    entry = next(x for x in instance.depth_distortions if x['axis']=='v' and x['layer']==layer and x['head']==head)
                    require(entry.get('value_joint_refit') is True and error == entry['squared_error_by_depth'][1], 'Post-V distortion not updated to physical fields')
        cells[(k,v)] = instance
        observations.append(dict(key_joint=k,value_joint=v,V_refit_heads=len(instance.value_joint_statistics),
            K_refit_heads=len(instance.joint_gain_statistics), stub_stored_field_bytes=physical_bytes(instance.states,book)))
    for k in (False,True):
        off,on = cells[(k,False)],cells[(k,True)]
        for layer,(a,b) in enumerate(zip(off.states,on.states)):
            require(torch.equal(a.direction_code,b.direction_code) and torch.equal(a.direction_gain,b.direction_gain), 'V flag changed K base fields')
            local = [(axis,head,stage) for axis,layer_id,head,stage in off.selected_residual_stages if layer_id==layer]
            for slot,(axis,_,_) in enumerate(local):
                if axis=='k': require(torch.equal(a.depth_residual_alpha[slot],b.depth_residual_alpha[slot]), 'V hook changed K alpha slot')
    for v in (False,True):
        off,on = cells[(False,v)],cells[(True,v)]
        for layer,(a,b) in enumerate(zip(off.states,on.states)):
            require(all(torch.equal(getattr(a,key),getattr(b,key)) for key in ('v_code','v_scale','v_min')), 'K flag changed V base fields')
            local = [(axis,head,stage) for axis,layer_id,head,stage in off.selected_residual_stages if layer_id==layer]
            for slot,(axis,_,_) in enumerate(local):
                if axis=='v': require(torch.equal(a.depth_residual_alpha[slot],b.depth_residual_alpha[slot]), 'K flag changed V alpha slot')
    require(len({x['stub_stored_field_bytes'] for x in observations})==1, 'Stored field bytes changed across cells')
    factory,executor = SimpleNamespace(bank_class=object()),SimpleNamespace(bank_class=object())
    old_factory,old_executor=factory.bank_class,executor.bank_class
    try:
        with scoped_registration(factory,executor,object()):
            cls.settings=variants[-1]; cls()._initialize(sources_k,sources_v,fail=True)
    except ValueError:
        pass
    require(factory.bank_class is old_factory and executor.bank_class is old_executor, 'Registration did not restore after failure')
    fixture=output/'postcheck_stub'; fixture.mkdir()
    (fixture/'manifest.json').write_text(json.dumps(dict(config_sha256='CPU-config',external_codec_inputs={})),encoding='utf-8')
    (fixture/'complete.json').write_text(json.dumps(dict(status='COMPLETE',records=4,expected=4)),encoding='utf-8')
    (fixture/'records.jsonl').write_text(''.join(json.dumps(dict(CPU_stub_record=i))+'\n' for i in range(4)),encoding='utf-8')
    sources=source_fingerprints()
    marker=finalize_run(fixture,{str(bookpath):sha(bookpath)},sources,'CPU-config',declarations)
    require(marker['manifest_sha256']==sha(fixture/'manifest.json') and marker['engine_complete_sha256']==sha(fixture/'complete.json') and marker['records_sha256']==sha(fixture/'records.jsonl'), 'Postcheck binding failed')
    return dict(passed=True, observations=observations, value_helper_CPU=value_controls,
        complete_owner_N1792_all_cells=plan['predicted_bytes'],ranking_bytes=448,full_priority=224,selected_stages=43,
        actual_stub_V_selected_heads=expected_v,actual_stub_K_selected_heads=expected_k,
        physical_slot_pointers_preserved=True,bits_signs_unchanged=True,orthogonal_KV_field_flags=True,
        updated_final_V_distortions=True,registration_exception_restore=True,postcheck_hash_binding=True,
        added_retained_tensor_bytes=0,
        scope='CPU synthetic physical codecs and28-layer stub orchestration; N1792 budget algebra separate from N32 stub; no model/task/timing measurement')


def gpu_controls():
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu requires actual CUDA; CPU is not a reader substitute')
    from fast_wht import torch_wht
    from round8_attention import attention
    from round7_key_attention import attention as frozen_attention
    torch.manual_seed(20261118);torch.backends.cuda.matmul.allow_tf32=False
    decoder,fold=frozen_reference_functions()
    n,hk,hq=64,4,28
    k=torch.randn(1,hk,n,128,device='cuda',dtype=torch.float16)
    v=torch.randn_like(k);v[0,1,:,0]=.125
    mean=k.float().mean(-2,keepdim=True).half()
    rotated=torch_wht(k.float()-mean.float())
    book=torch.randn(16,4,device='cuda',dtype=torch.float16)
    kc,kg=encode_key(rotated,book,gain_factor=torch.tensor([.91,.98,1.04,1.11],device='cuda')[None,:,None,None])
    rec=binary_centroids(v.transpose(2,3).contiguous(),group_size=n)
    vc,vs,vm=fold(rec,1,torch.tensor([.86,.95,1.06,1.14],device='cuda')[None,:,None,None],
        torch.tensor([-.1,.03,.12,-.06],device='cuda')[None,:,None,None])
    bk,bv=decode_key(kc,kg,book),decoder(vc,vs,vm,n,1).transpose(2,3).contiguous()
    kr,ka,_,_=encode_sign_depth(rotated[0,1],bk[0,1],1)
    vr,va,_,_=encode_sign_depth(v[0,1],bv[0,1],1)
    states,rows={},[]
    for kjoint in (False,True):
        for vjoint in (False,True):
            lookup=torch.full((2,hk,2),-1,device='cuda',dtype=torch.int32)
            lookup[0,1,0]=0;lookup[1,1,0]=1
            state=SimpleNamespace(direction_code=kc.clone(),direction_gain=kg.clone(),direction_mean=mean,
                v_code=vc.clone(),v_scale=vs.clone(),v_min=vm.clone(),depth_residual_lookup=lookup,
                depth_residual_code=torch.stack((kr[0],vr[0])),depth_residual_alpha=torch.stack((ka[0],va[0])))
            if kjoint:
                gain,alpha,_=joint_refit_key(rotated[0,1],key_template(state.direction_code[0,1],book),
                    state.direction_gain[0,1],state.depth_residual_code[0],state.depth_residual_alpha[0])
                state.direction_gain[0,1].copy_(gain);state.depth_residual_alpha[0].copy_(alpha)
            vst=None
            if vjoint:
                scale,minimum,alpha,vst=joint_refit_value(v[0,1],state.v_code[0,1],state.v_scale[0,1],state.v_min[0,1],
                    state.depth_residual_code[1],state.depth_residual_alpha[1],rounds=2)
                state.v_scale[0,1].copy_(scale);state.v_min[0,1].copy_(minimum);state.depth_residual_alpha[1].copy_(alpha)
                require(vst['max_finite_total_SSE_increase']==0,'CUDA V hard guard worsened total source SSE')
            require(torch.equal(state.v_code,vc) and torch.equal(state.depth_residual_code[0],kr[0]) and torch.equal(state.depth_residual_code[1],vr[0]),'CUDA code/sign changed')
            states[(kjoint,vjoint)]=state
            for nq in (1,11):
                prefix=41;nt=prefix+nq
                state.visual_pos=torch.arange(0,2*n,2,device='cuda')
                state.text_pos=torch.cat((torch.arange(1,2*prefix,2,device='cuda'),torch.arange(2*n,2*n+nq,device='cuda')))
                state.text_length,state.length=nt,2*n+nq
                # Repeated input seed makes the four cells share q/text fields.
                torch.manual_seed(20261118+nq)
                state.text_k=torch.randn(1,hk,nt+16,128,device='cuda',dtype=torch.float16)
                state.text_v=torch.randn_like(state.text_k)
                q=torch.randn(1,nq,hq,128,device='cuda',dtype=torch.float16).transpose(1,2)
                actual=attention(q,state,book,n)
                require(torch.equal(actual,frozen_attention(q,state,book,n)),'Updated physical fields differ across unchanged K4/binary readers')
                dense_k=decode_key(state.direction_code,state.direction_gain,book)
                dense_v=decoder(state.v_code,state.v_scale,state.v_min,n,1).transpose(2,3).contiguous()
                dense_k[0,1]=decode_sign_stage(dense_k[0,1],state.depth_residual_code[0],state.depth_residual_alpha[0])
                dense_v[0,1]=decode_sign_stage(dense_v[0,1],state.depth_residual_code[1],state.depth_residual_alpha[1])
                qr=torch_wht(q).half().float()
                zv=qr@dense_k.repeat_interleave(7,1).float().transpose(-1,-2)/128**.5
                zv+=(q.float()*mean.repeat_interleave(7,1).float()).sum(-1,keepdim=True)/128**.5
                zt=q.float()@state.text_k[:,:,:nt].repeat_interleave(7,1).float().transpose(-1,-2)/128**.5
                score=torch.cat((zv,zt),-1)
                positions=torch.cat((state.visual_pos,state.text_pos))
                qpos=torch.arange(state.length-nq,state.length,device='cuda')
                score.masked_fill_(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
                values=torch.cat((dense_v,state.text_v[:,:,:nt]),2).repeat_interleave(7,1)
                expected=(score.softmax(-1)@values.float()).half()
                delta=actual.float()-expected.float()
                maximum=float(delta.abs().max());relative=float(delta.norm()/expected.float().norm().clamp_min(1e-20))
                rows.append(dict(key_joint=kjoint,value_joint=vjoint,nv=n,nq=nq,query_heads=hq,kv_heads=hk,
                    max_abs=maximum,relative_l2=relative,passed=maximum<.003 and relative<.003,
                    frozen_reader_bitwise_equal=True,V_helper_stats=vst,
                    scope='actual unchanged reader vs dense current physical fields; no model/task/timing claim'))
    for kflag in (False,True):
        a,b=states[(kflag,False)],states[(kflag,True)]
        require(torch.equal(a.direction_gain,b.direction_gain) and torch.equal(a.depth_residual_alpha[0],b.depth_residual_alpha[0]),'CUDA V flag changed K fields')
    for vflag in (False,True):
        a,b=states[(False,vflag)],states[(True,vflag)]
        require(all(torch.equal(getattr(a,key),getattr(b,key)) for key in ('v_code','v_scale','v_min')) and torch.equal(a.depth_residual_alpha[1],b.depth_residual_alpha[1]),'CUDA K flag changed V fields')
    require(all(row['passed'] for row in rows),'Actual CUDA reader/dense tolerance failed')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sources_before=source_fingerprints()
    cpu=cpu_controls(args.output)
    with torch.inference_mode():
        gpu=gpu_controls() if args.gpu else []
    require(source_fingerprints()==sources_before,'Sources changed during check')
    report=dict(status='PASS',CPU_controls=cpu,GPU_controls=gpu,gpu_executed=bool(args.gpu),
        source_sha256=sources_before,checker_sha256=sha(__file__),
        check_support_sha256={name:sha(Path(__file__).with_name(name)) for name in ('round8_check.py','round7_key_attention.py')},
        scope='synthetic same-representation arithmetic and stub integration only; no full model/quality/speed evidence')
    (args.output/'check.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',gpu_executed=bool(args.gpu),
        CPU_factorial_cells=4,GPU_reader_cases=len(gpu),report_sha256=sha(args.output/'check.json'))),encoding='utf-8')
    print(json.dumps(dict(status='PASS',GPU_executed=bool(args.gpu),CPU_factorial_cells=4,GPU_reader_cases=len(gpu))),flush=True)


if __name__=='__main__':
    main()
