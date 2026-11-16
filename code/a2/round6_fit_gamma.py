"""Fixed-mask F16 residual amplitude fitting with an exact packed forward.

The local objective uses the same compressed student q and text cache for both
visual representations and compares total attention readouts after W_O. Final
KL/CE objectives backpropagate through the complete compressed question replay.
All objectives fit the same [2,L,Hkv] family for the same scheduled cases/steps.
Only the dense derivative is surrogate; every forward is the deployed reader.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/experiments'))
from bank import install_legacy
from round6_student_codec import (sha, selected_rows, settings_from_config, select_units,
    encode_layer, signs, teacher_case, make_prefix_cache, mask_policy_signature)
from fast_wht import torch_wht
from mixed_residual_attention import attention as packed_attention


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    a = c['arguments']
    install_legacy(a['legacy_scripts'])
    from p180_multiturn_core import attention_bridge
    from p184_working_set import attention
    import run_p175_modern_replication as oldbase
    from transformers import AutoProcessor, AutoModelForImageTextToText
    objective_name = c.get('objective', 'full_model_KL')
    if objective_name not in ('local_functional', 'full_model_KL', 'full_model_CE'):
        raise ValueError('objective must be local_functional, full_model_KL or full_model_CE')
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = settings_from_config(c)
    base = json.loads(Path(settings['metadata_factor_path']).read_text())
    if base['encoder_book_sha256'] != sha(settings['direction_book_path']) or base['value_group_size'] != settings['v_group_size']:
        raise ValueError('Frozen book/base/window mismatch')
    rows = selected_rows(json.loads(Path(a['samples']).read_text(encoding='utf-8')),
        int(c.get('calibration_image_count', 16)), int(c.get('max_cases', 16)), bool(c.get('first_question_per_image', True)))
    if not rows:
        raise ValueError('No calibration cases')
    processor = AutoProcessor.from_pretrained(a['model'], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(a['model'], local_files_only=True,
        dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config.get_text_config()
    nl, nh = config.num_hidden_layers, config.num_key_value_heads
    modules = [layer.self_attn for layer in model.model.language_model.layers]
    book = torch.tensor(json.loads(Path(settings['direction_book_path']).read_text())['centers'], device=model.device, dtype=torch.float16)
    gamma_bytes = 2 * nl * nh * 2
    cases = []
    for row in rows:
        case = teacher_case(model, processor, row, a, int(c.get('target_tokens', 4)), attention_bridge, oldbase.stop_ids(model, processor))
        visual = case['visual']
        selected, budget = select_units(settings, base, nl, nh, visual.numel(), gamma_bytes)
        vg = visual.numel() if settings['v_group_size'] == 'all' else int(settings['v_group_size'])
        case['encoded'] = [encode_layer(k.index_select(2, visual), v.index_select(2, visual), book, base, i, vg)
                           for i, (k, v) in enumerate(case['prefix'])]
        case['selected'], case['budget'] = selected, budget
        cases.append(move(case, 'cpu'))
        print(json.dumps({'phase':'gamma_teacher_collection','case':len(cases),'cases':len(rows),'id':row['id'],
            'visual_tokens':visual.numel(),'selected_units':len(selected),'actual_bits':budget['actual_bits']}), flush=True)
        del case, visual
    log_gamma = torch.nn.Parameter(torch.zeros((2, nl, nh), device=model.device, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_gamma], lr=float(c.get('learning_rate', .003)))
    forward_controls = {'calls': 0, 'exact_equal': 0, 'max_abs': 0.}

    def loss_for(cpu_case):
        case = move(cpu_case, model.device)
        visual, encoded, selected = case['visual'], case['encoded'], set(case['selected'])
        cache, _, _ = make_prefix_cache(case['prefix'], config, False)
        suffix, qpos = case['suffix'], case['qpos']
        total = case['boundary'] + suffix.shape[-1]
        allpos = torch.arange(total, device=model.device)
        keep = torch.ones(total, device=model.device, dtype=torch.bool)
        keep[visual] = False
        textpos = allpos[keep]
        positions = torch.cat((visual, textpos))
        local_losses = []
        # Both casts are in the training forward and runtime: F16 gamma table,
        # then F16(gamma_f16 * original_alpha_f16). No hidden FP32 gamma runtime.
        gamma = log_gamma.exp().half()

        def read(q, index):
            entry = encoded[index]
            group = q.shape[1] // nh
            kr, vv = entry['krot'], entry['v_base']
            heads_k, heads_v, compact_codes, compact_alphas = [], [], [], []
            lookup = torch.full((2, nh), -1, device=q.device, dtype=torch.int32)
            for axis, original, heads in (('k', kr, heads_k), ('v', vv, heads_v)):
                sg = signs(entry[axis+'_sign'], original.shape)
                for head in range(nh):
                    value = original[:, head:head+1]
                    if (axis, index, head) in selected:
                        alpha = (entry[axis+'_alpha'][:, head:head+1].float() * gamma[0 if axis=='k' else 1,index,head].float()).half()
                        value = (value.float() + sg[:,head:head+1] * alpha.float()[...,None]).half()
                        lookup[0 if axis=='k' else 1,head] = len(compact_codes)
                        compact_codes.append(entry[axis+'_sign'][0,head])
                        compact_alphas.append(alpha[0,0])
                    heads.append(value)
            rotated = torch.cat(heads_k,1).repeat_interleave(group,1)
            decoded_v = torch.cat(heads_v,1).repeat_interleave(group,1)
            qdense = q.detach() if objective_name == 'local_functional' else q
            qr = torch_wht(qdense).half().float()
            score_v = qr @ rotated.float().transpose(-1,-2) / 128**.5
            score_v = score_v + (qdense.float()*entry['mean'].repeat_interleave(group,1).float()).sum(-1,keepdim=True) / 128**.5
            key = cache.layers[index].keys.index_select(2,textpos).repeat_interleave(group,1)
            value = cache.layers[index].values.index_select(2,textpos).repeat_interleave(group,1)
            if objective_name == 'local_functional':
                key, value = key.detach(), value.detach()
            score_t = qdense.float() @ key.float().transpose(-1,-2) / 128**.5
            scores = torch.cat((score_v,score_t),-1)
            mask = positions[None,None,None,:] <= qpos[:,None,:,None]
            scores = scores.masked_fill(~mask,-torch.inf)
            weights = (scores-scores.amax(-1,keepdim=True)).exp()
            values = torch.cat((decoded_v,value),2)
            dense = ((weights.half().float() @ values.float())/weights.sum(-1,keepdim=True)).half()
            with torch.no_grad():
                nt = textpos.numel()
                capacity = int(c.get('text_capacity',512))
                if nt > capacity:
                    raise ValueError('Increase explicit text_capacity for this calibration prompt')
                tk = torch.empty((1,nh,capacity,128),device=q.device,dtype=torch.float16)
                tv = torch.empty_like(tk)
                tk[:,:,:nt].copy_(cache.layers[index].keys.index_select(2,textpos))
                tv[:,:,:nt].copy_(cache.layers[index].values.index_select(2,textpos))
                state = SimpleNamespace(direction_code=entry['codes'],direction_gain=entry['gain'],direction_mean=entry['mean'],
                    v_code=entry['v_record'][0],v_scale=entry['v_record'][1],v_min=entry['v_record'][2],
                    mixed_residual_lookup=lookup,text_k=tk,text_v=tv,text_pos=textpos,visual_pos=visual,text_length=nt,length=total)
                if compact_codes:
                    state.mixed_residual_code = torch.stack(compact_codes).contiguous()
                    state.mixed_residual_alpha = torch.stack(compact_alphas).detach().contiguous()
                exact = packed_attention(q,state,book,entry['value_group'])
            output = (dense.float()+(exact.float()-dense.float()).detach()).half()
            difference = float((output.detach().float()-exact.float()).abs().max())
            forward_controls['calls'] += 1
            forward_controls['exact_equal'] += int(torch.equal(output.detach(),exact))
            forward_controls['max_abs'] = max(forward_controls['max_abs'],difference)
            if difference or not torch.equal(output.detach(),exact):
                raise RuntimeError('Gamma surrogate changed actual packed forward')
            if objective_name == 'local_functional':
                # Same student q and text states; total readout includes both
                # segments. Reference is detached so only encoder amplitudes fit.
                with torch.no_grad():
                    full_k = case['prefix'][index][0].index_select(2,visual).repeat_interleave(group,1)
                    full_v = case['prefix'][index][1].index_select(2,visual).repeat_interleave(group,1)
                    reference_scores = torch.cat((qdense.float()@full_k.float().transpose(-1,-2)/128**.5,score_t.detach()),-1).masked_fill(~mask,-torch.inf)
                    reference = reference_scores.softmax(-1) @ torch.cat((full_v,value.detach()),2).float()
                n = case['targets'].shape[1]
                delta = (output[:,:,-n:].float()-reference[:,:,-n:]).transpose(1,2).reshape(1,n,-1)
                projected = F.linear(delta,modules[index].o_proj.weight.float())
                local_losses.append(projected.square().mean())
                return exact.detach()
            return output
        for index,module in enumerate(modules):
            module._p185_read = lambda q,index=index: read(q,index)
        try:
            with attention(model):
                out = model(input_ids=suffix,position_ids=qpos,past_key_values=cache,use_cache=True,
                            return_dict=True,logits_to_keep=case['targets'].shape[1])
        finally:
            for module in modules:
                del module._p185_read
        if objective_name == 'local_functional':
            # Layers without active residuals have a constant local error; omit
            # only their constant contribution, retaining equal layer scaling.
            return sum(local_losses) / nl
        if objective_name == 'full_model_CE':
            return F.cross_entropy(out.logits.float().reshape(-1,out.logits.shape[-1]),case['targets'].reshape(-1))
        target = case['teacher_logits'].float().log_softmax(-1)
        lp = out.logits.float().log_softmax(-1)
        return (target.exp()*(target-lp)).sum(-1).mean()

    manifest = {'schema':'trajkv_round6_gamma_fit_v1','config':c,'config_sha256':sha(args.config),
        'samples_sha256':sha(a['samples']),'encoder_book_sha256':sha(settings['direction_book_path']),
        'base_metadata_sha256':sha(settings['metadata_factor_path']),'calibration_ids':[r['id'] for r in rows],
        'incumbent_mask_policy_sha256':mask_policy_signature(settings),
        'labels_used':False,'model_weights_frozen':True,'base_book_mask_frozen':True,
        'parameter_shape':[2,nl,nh],'parameter_count':log_gamma.numel(),'encoder_constant_dtype':'float16',
        'encoder_constant_bytes':gamma_bytes,'objective':objective_name,'packed_forward':True,
        'local_scope':'same compressed student q/text; full-softmax visual+text output after W_O',
        'compression_boundary':'after last visual token before complete first-question suffix',
        'sources':{x.name:sha(x) for x in (Path(__file__),Path(__file__).with_name('round6_student_codec.py'),Path(__file__).with_name('round6_gamma_bank.py'))}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary = int(c.get('summary_cases',2))
    with torch.no_grad():
        initial = [float(loss_for(case)) for case in cases[:summary]]
    logs=[]
    scale = float(c.get('gradient_loss_scale',128.))
    for step in range(int(c.get('steps',16))):
        started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        case=cases[step%len(cases)]
        loss=loss_for(case)
        target=loss+float(c.get('regularization',.0))*log_gamma.square().mean()
        if not target.requires_grad or not bool(torch.isfinite(target)):
            raise RuntimeError('Detached/nonfinite gamma objective')
        (target*scale).backward()
        if log_gamma.grad is None:
            raise RuntimeError('Missing gamma gradient')
        log_gamma.grad.div_(scale)
        if not bool(torch.isfinite(log_gamma.grad).all()):
            raise RuntimeError('Nonfinite gamma gradient')
        grad=float(log_gamma.grad.norm())
        optimizer.step()
        with torch.no_grad():
            bound=float(c.get('gamma_log_bound',.7))
            log_gamma.clamp_(-bound,bound)
        torch.cuda.synchronize()
        entry={'step':step,'id':case['id'],'objective':objective_name,'loss_before_update':float(loss.detach()),
            'gradient_norm':grad,'seconds':time.perf_counter()-started}
        logs.append(entry);print(json.dumps(entry),flush=True)
        del loss,target
    with torch.no_grad():
        final=[float(loss_for(case)) for case in cases[:summary]]
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Model weights were modified')
    result={'schema':'trajkv_round6_gamma_v1','status':'COMPLETE','gamma_f16':log_gamma.detach().exp().half().cpu().tolist(),
        'encoder_book_sha256':manifest['encoder_book_sha256'],'base_metadata_sha256':manifest['base_metadata_sha256'],
        'incumbent_mask_policy_sha256':manifest['incumbent_mask_policy_sha256'],
        'objective':objective_name,'cases':len(cases),'images':len({r['image_id'] for r in rows}),
        'steps':len(logs),'calibration_ids':manifest['calibration_ids'],'parameter_count':log_gamma.numel(),
        'encoder_constant_bytes':gamma_bytes,'encoder_constant_dtype':'float16','packed_forward':True,
        'forward_identity_controls':forward_controls,'initial_objective':initial,'final_objective':final,
        'steps_log':logs,'per_case_budget':[case['budget'] for case in cases],
        'labels_used':False,'model_weights_frozen':True,'fixed_base_book_mask':True,
        'training_forward_rounding':'gamma=F16(exp(loggamma)); alpha=F16(F32(gamma)*F32(original_alpha)); exact packed reader',
        'serving_scope':'gamma folded into existing F16 alpha at encode time; same reader; no retained source visual KV',
        'scope':'offline calibration objective only; actual packed free answers required for quality claims'}
    (args.output/'factors.json').write_text(json.dumps(result,indent=2))
    (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','cases':len(cases),'steps':len(logs)}))


if __name__=='__main__':
    main()
