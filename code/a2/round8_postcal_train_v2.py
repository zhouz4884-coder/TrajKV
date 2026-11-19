"""v2 final-R42 KL calibration with explicit finite-zero gradient semantics.

Frozen model/book/f0/J/sign; absolute FP32 post[3,28,4] is the only learned
artifact. Every deployed forward uses the unchanged packed reader. Dense
attention supplies a surrogate derivative, never an exact quantized gradient.
No select/GT input or checkpoint selection is supported.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=('round8_postcal_train_v2.py','round8_postcal_fields.py','round8_postcal_schedule.py',
    'round8_bank.py','round8_codec.py','round8_attention.py','round8_joint_gain.py',
    'round6_fit_gamma.py','round6_student_codec.py','calibrate_gradient_rank_round5.py',
    'round6_depth_codec.py','round7_key_layout.py','round7_key_artifacts.py','masked_bank.py',
    'metadata_direction_bank.py','learned_direction_bank.py','quantizers.py','affine_rate_bank.py',
    'fast_wht.py','session_boundary.py','round8_continuous.py')
FIELDS=('direction_code','direction_gain','direction_mean','v_code','v_scale','v_min',
    'depth_residual_lookup','depth_residual_code','depth_residual_alpha','visual_pos')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def require(value,message):
    if not value:raise RuntimeError(message)


def source_fingerprints():
    result={name:sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    for name in ('bank.py','fused_progressive.py'):
        result['experiments/'+name]=sha(ROOT/'code/experiments'/name)
    return result


def validate_config(config):
    require(config.get('schema')=='trajkv.round8.postcal-train.v1','Explicit postcal training schema required')
    require(config.get('steps')==200 and config.get('seed')==1212,'Freeze200 steps and seed1212')
    require(config.get('objective')=='full_teacher_KL' and config.get('checkpoint')=='final_step200',
        'Only fixed-final-step Full teacher KL; no select/checkpoint tuning')
    require(not any(key in config for key in ('references','select','select_samples','GT','labels')),'No GT/select input supported')
    args=config['arguments'];s=config['bank_settings']
    require(args['max_patches']==s['cap']==8 and args.get('compress_before_question') is True,'Explicit image-boundary cap8 required')
    require(s['method']=='round8_joint' and s.get('key_assignment','l2')=='l2' and s['key_amplitude']=='norm'
        and s['key_encoding']=='reencode' and s['value_encoding']=='binary' and s['v_group_size']=='all','Old K1/binary/all/L2/norm required')
    require(s.get('joint_gain_refit',False) is False and s.get('value_joint_refit',False) is False
        and s.get('postres_direction',False) is False and s.get('final_ray_mode','off')=='off'
        and s.get('visual_mean_correction',False) is False,'No other geometry intervention in this postcal family')
    ranking=s['selected_stages'];require(len(ranking)==224 and len({tuple(item) for item in ranking})==224
        and all(a in ('k','v') and 0<=l<28 and 0<=h<4 and d==0 for a,l,h,d in ranking)
        and s['res_stages_limit']==42 and ranking[42]==['v',27,3,0],'Fixed full224 and original42 prefix required')
    require(sum(item[0]=='k' for item in ranking[:42])==23 and sum(item[0]=='v' for item in ranking[:42])==19,'Expected23K19V R42')
    require(s['direction_book_path']==s['metadata_reference_book_path'] and s['round8_factor_policy']=='inherit_from_reference_book',
        'Old exact book/f0 numerical inheritance required')
    require(config.get('target_tokens')==8 and config.get('text_capacity',512)>=512,'Fixed8 teacher tokens and explicit text capacity')
    require(config.get('parameter_shape')==[3,28,4] and config.get('encoder_constant_bytes')==1344,'Actual post336 FP32 price required')
    from round6_depth_codec import _pack_ids
    full_priority_sha256=hashlib.sha256(_pack_ids(ranking,28,4,True)).hexdigest()
    selected42_sha256=hashlib.sha256(_pack_ids(ranking[:42],28,4,True)).hexdigest()
    contract=dict(stage_count=42,K_stages=23,V_stages=19,full_priority_sha256=full_priority_sha256,full_priority_bytes=448,
        full_priority_JSON_sha256=canonical(ranking),selected42_sha256=selected42_sha256,selected42_packed_bytes=84,
        selected42_JSON_sha256=canonical(ranking[:42]),old43_JSON_sha256=canonical(ranking[:43]),
        new_encoder_bytes=1344,added_retained_decoder_tensor_bytes=0)
    if config.get('priority_contract') is not None:require(config['priority_contract']==contract,'Explicit packed/JSON priority contract differs')
    require(config.get('gradient_gate_policy')==dict(probe_nonzero=True,optimizer_nonzero_steps=[0],later_finite_zero_allowed=True,missing_or_nonfinite_fatal=True),'Explicit v2 gradient gate policy required')
    return contract


def input_fingerprints(config_path,config,*,verify_images=True):
    paths={str(config_path),config['arguments']['samples'],config['bank_settings']['direction_book_path'],
        config['bank_settings']['metadata_factor_path'],config['bank_settings']['metadata_reference_book_path'],
        str(Path(config['arguments']['model'])/'config.json'),config['arguments']['kivi_source']}
    legacy=Path(config['arguments']['legacy_scripts'])
    require(legacy.is_dir(),'Legacy runtime directory unavailable')
    paths.update(str(p) for p in legacy.glob('*.py'))
    image_expected={}
    if verify_images:
        for row in read(config['arguments']['samples']):
            require(not any(k in row for k in ('answer','answers','references')),'Calibration input must be answer-free')
            for path,h in zip(row['remote_images'],row['hashes']):image_expected[path]=h;paths.add(path)
    result={path:sha(path) for path in sorted(paths)}
    require(all(result[p]==h for p,h in image_expected.items()),'Calibration image bytes differ from frozen sample SHA')
    return result


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(move(v,device) for v in value)
    return value


def dense_readout(q,krot,value,mean,text_k,text_v,visual_pos,text_pos,total,wht):
    """Same rounded fields/full softmax, with the existing dense STE derivative."""
    groups=q.shape[1]//krot.shape[1];d=q.shape[-1]
    qr=wht(q).half().float()
    zv=qr@krot.repeat_interleave(groups,1).float().transpose(-1,-2)/d**.5
    zv=zv+(q.float()*mean.repeat_interleave(groups,1).float()).sum(-1,keepdim=True)/d**.5
    zt=q.float()@text_k.repeat_interleave(groups,1).float().transpose(-1,-2)/d**.5
    score=torch.cat((zv,zt),-1);positions=torch.cat((visual_pos,text_pos))
    qpos=torch.arange(total-q.shape[-2],total,device=q.device)
    score=score.masked_fill(positions[None,None,None,:]>qpos[None,None,:,None],-torch.inf)
    weight=(score-score.amax(-1,keepdim=True)).exp()
    values=torch.cat((value,text_v),2).repeat_interleave(groups,1)
    return ((weight.half().float()@values.float())/weight.sum(-1,keepdim=True)).half()


def bridge_forward(dense,exact):
    result=(dense.float()+(exact.float()-dense.float()).detach()).half()
    require(torch.equal(result.detach(),exact),'Dense STE bridge changed actual packed output')
    return result


def state_fingerprint(state):
    return {name:dict(sha256=tensor_sha(getattr(state,name)),shape=list(getattr(state,name).shape),dtype=str(getattr(state,name).dtype))
        for name in FIELDS if hasattr(state,name)}


def collect_old_R42(case,model,config,book,settings):
    """Use the actual old bank under no_grad; never reuse sampled native dumps."""
    from round8_bank import Round8Bank
    from round6_student_codec import make_prefix_cache
    cache,_,_=make_prefix_cache(case['prefix'],config,False)
    previous=Round8Bank.settings;Round8Bank.settings=settings;bank=None
    try:
        with torch.no_grad():
            bank=Round8Bank(model,cache,case['visual'],'round8_joint',case['legacy_scripts'],case['kivi_source'],False)
            require(len(bank.states)==28 and bank.selected_residual_stages==[tuple(x) for x in settings['selected_stages'][:42]],'Production R42 encoding mask differs')
            encoded=[]
            for state in bank.states:
                encoded.append({name:getattr(state,name).detach().cpu().clone() for name in FIELDS if hasattr(state,name)})
            nv=int(case['visual'].numel());owner=bank.owned();post_bytes=owner['complete_visual_owner_bytes']+1344
            old43_bytes=owner['complete_visual_owner_bytes']+18*nv
            require(post_bytes<=old43_bytes and nv>=96,'R42+1344B exceeds old actual R43 B0')
            require(bank.book.shape==(16,4) and torch.equal(bank.book,book),'Actual old book differs')
            budget=dict(real_Nv=nv,old_R42_owner=owner,post_encoder_bytes=1344,post_R42_complete_bytes=post_bytes,
                old_R43_complete_B0=old43_bytes,delta_to_old_B0=1344-18*nv,post_actual_bits=16*post_bytes/bank.visual_full_bytes)
        return encoded,budget
    finally:
        if bank is not None:bank.close()
        Round8Bank.settings=previous


class PackedKLLoss:
    """The model forward is exact packed; only its derivative is the surrogate."""
    def __init__(self,model,config,book,parameters,options):
        self.model,self.config,self.book,self.parameters,self.options=model,config,book,parameters,options
        self.modules=[layer.self_attn for layer in model.model.language_model.layers]
        self.controls=dict(packed_calls=0,bridge_bitwise_equal=0,identity_reader_calls=0,identity_reader_equal=0,
            identity_field_layers=0,shadow_max_abs=0.,shadow_max_relative_l2=0.,q_gradient_calls=0,q_gradient_squared_norm=0.,q_gradient_finite=True)
        self.step_q_count=0;self.step_q_squared=0.;self.step_q_finite=True

    def table(self):
        return torch.stack((self.parameters[0].exp(),self.parameters[1].exp(),self.parameters[2]))

    def reset_step_q(self):self.step_q_count=0;self.step_q_squared=0.;self.step_q_finite=True

    def q_hook(self,gradient):
        finite=bool(torch.isfinite(gradient).all());norm=float((gradient.float()/128).square().sum())
        self.step_q_count+=1;self.step_q_squared+=norm;self.step_q_finite&=finite
        self.controls['q_gradient_calls']+=1;self.controls['q_gradient_squared_norm']+=norm;self.controls['q_gradient_finite']&=finite
        return gradient

    def __call__(self,cpu_case,*,identity=False):
        from round6_student_codec import make_prefix_cache
        from round8_postcal_fields import postcal_fields,decode_postcal_visual_fields
        from round8_attention import attention as packed_attention
        from fast_wht import torch_wht
        from p184_working_set import attention
        case=move(cpu_case,self.model.device);cache,_,_=make_prefix_cache(case['prefix'],self.config,False)
        visual=case['visual'];suffix=case['suffix'];total=case['boundary']+suffix.shape[-1]
        allpos=torch.arange(total,device=self.model.device);keep=torch.ones(total,device=self.model.device,dtype=torch.bool);keep[visual]=False
        textpos=allpos[keep];table=self.table();capacity=int(self.options.get('text_capacity',512));nh=4
        require(textpos.numel()<=capacity,'Explicit text capacity insufficient; preserve this run')
        def reader(q,index):
            base=SimpleNamespace(**case['encoded'][index]);fields=postcal_fields(base,table[:,index])
            krot,value=decode_postcal_visual_fields(base,table[:,index],self.book,int(visual.numel()))
            text_k=cache.layers[index].keys.index_select(2,textpos);text_v=cache.layers[index].values.index_select(2,textpos)
            if torch.is_grad_enabled() and q.requires_grad:q.register_hook(self.q_hook)
            dense=dense_readout(q,krot,value,fields.direction_mean,text_k,text_v,visual,textpos,total,torch_wht)
            with torch.no_grad():
                nt=textpos.numel();tk=torch.empty((1,nh,capacity,128),device=q.device,dtype=torch.float16);tv=torch.empty_like(tk)
                tk[:,:,:nt].copy_(text_k);tv[:,:,:nt].copy_(text_v)
                fields.text_k,fields.text_v,fields.text_pos=tk,tv,textpos
                fields.visual_pos,fields.text_length,fields.length=visual,nt,total
                exact=packed_attention(q,fields,self.book,int(visual.numel()))
                if identity:
                    for name in FIELDS:
                        if hasattr(base,name):require(torch.equal(getattr(base,name),getattr(fields,name)),'Theta0 changed old R42 physical field: '+name)
                    self.controls['identity_field_layers']+=1
                    old=SimpleNamespace(**vars(base),text_k=tk,text_v=tv,text_pos=textpos,text_length=nt,length=total)
                    old.visual_pos=visual;reference=packed_attention(q,old,self.book,int(visual.numel()))
                    self.controls['identity_reader_calls']+=1
                    require(torch.equal(reference,exact),'Theta0 actual packed output differs from old R42')
                    self.controls['identity_reader_equal']+=1
                delta=dense.detach().float()-exact.float();maximum=float(delta.abs().max())
                relative=float(delta.norm()/exact.float().norm().clamp_min(1e-20))
                abs_tol=float(self.options.get('shadow_absolute_tolerance',.003));rel_tol=float(self.options.get('shadow_relative_tolerance',.003))
                require(maximum<=abs_tol+rel_tol*float(exact.float().abs().max()) and relative<=rel_tol,'Dense shadow vs same physical packed reader exceeds declared tolerance')
                self.controls['shadow_max_abs']=max(self.controls['shadow_max_abs'],maximum)
                self.controls['shadow_max_relative_l2']=max(self.controls['shadow_max_relative_l2'],relative)
            result=bridge_forward(dense,exact);self.controls['packed_calls']+=1;self.controls['bridge_bitwise_equal']+=1
            return result
        require(not any(hasattr(module,'_p185_read') for module in self.modules),'Stale attention hooks before calibration loss')
        for index,module in enumerate(self.modules):module._p185_read=lambda q,index=index:reader(q,index)
        try:
            with attention(self.model):
                out=self.model(input_ids=suffix,position_ids=case['qpos'],past_key_values=cache,use_cache=True,
                    return_dict=True,logits_to_keep=case['targets'].shape[1])
        finally:
            for module in self.modules:
                if hasattr(module,'_p185_read'):del module._p185_read
        teacher=case['teacher_logits'].float().log_softmax(-1);student=out.logits.float().log_softmax(-1)
        loss=(teacher.exp()*(teacher-student)).sum(-1).mean()
        require(bool(torch.isfinite(loss)),'Nonfinite Full-teacher KL')
        return loss


def aggregate_kl(loss,cases,weights,*,identity=False):
    with torch.no_grad():values=[float(loss(case,identity=identity)) for case in cases]
    require(len(values)==len(weights) and abs(sum(weights)-1)<1e-9,'Source-balanced objective weight mismatch')
    return dict(per_case_KL=values,source_balanced_mean=sum(a*b for a,b in zip(values,weights)),case_count=len(values),weights=weights)


def gradient_diagnostic(step,norms,present,finite,q_count,q_finite,q_squared):
    """Reject missing/nonfinite; only the mandatory first step needs nonzero."""
    require(all(present),'Missing post parameter gradient')
    require(all(finite) and all(math.isfinite(v) for v in norms),'Nonfinite post parameter gradient')
    require(q_count>0,'Missing differentiable student query hook')
    require(q_finite and math.isfinite(q_squared),'Nonfinite student query gradient')
    if step==0:
        require(all(v>0 for v in norms) and q_squared>0,'First optimizer step requires nonzero q/all post-family gradients')
    return dict(parameter_zero_families=[index for index,value in enumerate(norms) if value==0.],
        query_finite_zero=q_squared==0.,all_parameter_families_finite=True,query_hook_present=True,
        query_gradient_finite=True,accepted_finite_zero=(q_squared==0. or any(v==0. for v in norms)))


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {key:finite_json(item) for key,item in value.items()}
    if isinstance(value,list):return [finite_json(item) for item in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    cfg=read(a.config);contract=validate_config(cfg)
    from round8_postcal_schedule import build_schedule
    if a.plan_only:
        from round8_quality_owner import local_path
        data=read(local_path(cfg['arguments']['samples']));flat,schedule,meta=build_schedule(data,steps=200,seed=1212)
        print(json.dumps(dict(status='PLAN_ONLY',config_sha256=sha(a.config),contract=contract,source_sha256=source_fingerprints(),
            sources=len(data),questions=len(flat),steps=len(schedule),schedule=meta,GPU_executed=False,model_forward_executed=False,
            required_final_marker='complete_postcal_train.json',new_encoder_constant_bytes=1344)),flush=True);return
    a.output.mkdir(parents=True,exist_ok=False);before=source_fingerprints();inputs=None
    try:
        require(torch.cuda.is_available(),'Actual CUDA mandatory for model calibration; no CPU substitute')
        torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        sys.path.insert(0,str(ROOT/'code/experiments'))
        from bank import install_legacy
        args=cfg['arguments'];install_legacy(args['legacy_scripts'])
        from p180_multiturn_core import attention_bridge
        import run_p175_modern_replication as oldbase
        from round6_student_codec import teacher_case
        from transformers import AutoProcessor,AutoModelForImageTextToText
        inputs=input_fingerprints(a.config,cfg);flat,schedule,meta=build_schedule(read(args['samples']),steps=200,seed=1212)
        require(len(flat)==70 and len(meta['flat_question_objective_weights'])==70,'Exact32sources/70 independent questions required')
        write_new(a.output/'schedule.json',dict(schedule=schedule,metadata=meta,flat_QA_ids=[row['id'] for row in flat]))
        processor=AutoProcessor.from_pretrained(args['model'],local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(args['model'],local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        model_config=model.config.get_text_config();require(model_config.num_hidden_layers==28 and model_config.num_key_value_heads==4,'Main28x4 model geometry required')
        book=torch.tensor(read(cfg['bank_settings']['direction_book_path'])['centers'],device=model.device,dtype=torch.float16)
        require(book.shape==(16,4),'Old exact globalGaussian16 book required')
        f0=read(cfg['bank_settings']['metadata_factor_path']);require(f0['key_gain_mode']=='centered','Freeze original centered f0')
        parameters=[torch.nn.Parameter(torch.zeros((28,4),device=model.device,dtype=torch.float32)) for _ in range(3)]
        loss=PackedKLLoss(model,model_config,book,parameters,cfg)
        cases=[];contexts=[]
        for index,row in enumerate(flat):
            case=teacher_case(model,processor,row,args,8,attention_bridge,oldbase.stop_ids(model,processor))
            require(case['id']==row['id'],'Independent QA/teacher case identity mismatch')
            case['legacy_scripts'],case['kivi_source']=args['legacy_scripts'],args['kivi_source']
            encoded,budget=collect_old_R42(case,model,model_config,book,cfg['bank_settings']);case['encoded']=encoded;case['budget']=budget
            contexts.append(dict(id=row['id'],image_id=row['image_id'],target_tokens=case['targets'].numel(),teacher_ids=case['targets'].cpu().tolist(),
                teacher_logits_sha256=tensor_sha(case['teacher_logits']),boundary=case['boundary'],prompt_suffix_tokens=case['prompt_suffix_tokens'],
                source_real_Nv=int(case['visual'].numel()),encoded_layers=[state_fingerprint(SimpleNamespace(**entry)) for entry in encoded],budget=budget))
            cases.append(move(case,'cpu'));print(json.dumps(dict(phase='teacher_and_R42_collection',case=index+1,cases=70,id=row['id'],Nv=budget['real_Nv'])),flush=True)
            del case
            if index==0:
                # A bounded actual first-Q gate precedes the remaining teacher
                # collection and all optimizer updates. No probe parameter is
                # updated; final200-step schedule remains exactly predeclared.
                started=time.perf_counter();loss.reset_step_q();probe_loss=loss(cases[0],identity=True)
                (probe_loss*128).backward();norms=[]
                for parameter in parameters:
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),'Actual first probe missing/nonfinite post gradient')
                    norms.append(float(parameter.grad.float().norm()/128));parameter.grad=None
                require(all(norm>0 for norm in norms) and loss.step_q_count>0 and loss.step_q_finite and loss.step_q_squared>0,
                    'Actual first-Q probe lacks nonzero finite q/post gradients')
                require(all(not p.requires_grad and p.grad is None for p in model.parameters()),'Actual first probe changed model gradient state')
                torch.cuda.synchronize();probe=dict(status='PASS',id=row['id'],real_Nv=budget['real_Nv'],actual_cap=8,
                    actual_packed_zero_theta_identity=True,dense_shadow_tolerance_pass=True,parameter_gradient_norms=norms,
                    q_gradient_calls=loss.step_q_count,q_gradient_norm=loss.step_q_squared**.5,
                    KL=float(probe_loss.detach()),seconds=time.perf_counter()-started,optimizer_updates=0,model_weights_frozen=True)
                write_new(a.output/'first_probe.json',probe);print(json.dumps(dict(phase='actual_first_gradient_probe',**probe)),flush=True);del probe_loss
        write_new(a.output/'teacher_contexts.json',dict(cases=contexts,labels_used=False,target_source='Full greedy first<=8tokens; no GT',scope='teacher logits pinned by SHA, recomputable from fixed model/source; no tensor cache artifact'))
        optimizer=torch.optim.Adam(parameters,lr=float(cfg.get('learning_rate',.003)))
        manifest=dict(schema='trajkv.round8.postcal-train-manifest.v1',config=cfg,config_sha256=sha(a.config),source_sha256=before,input_sha256=inputs,
            samples_sha256=sha(args['samples']),old_f0_sha256=sha(cfg['bank_settings']['metadata_factor_path']),encoder_book_sha256=sha(cfg['bank_settings']['direction_book_path']),
            contract=contract,schedule_sha256=sha(a.output/'schedule.json'),teacher_contexts_sha256=sha(a.output/'teacher_contexts.json'),
            first_actual_probe_sha256=sha(a.output/'first_probe.json'),
            labels_used=False,model_weights_frozen=True,optimization_parameter_shape=[3,28,4],post_encoder_constant_bytes=1344,
            compressed_question_replay=True,objective='Full teacher-to-student KL; tokenmean perQ/sourcebalanced aggregate',
            gradient_scope='actual packed forward and dense STE surrogate derivative through student q/text; not exact quantized or unbiased full-model gradient')
        write_new(a.output/'manifest.json',manifest)
        weights=meta['flat_question_objective_weights'];initial=aggregate_kl(loss,cases,weights,identity=True);write_new(a.output/'initial_kl.json',initial)
        logs=[];zero_steps=[];zero_q_steps=[];zero_family_counts=[0,0,0]
        with (a.output/'steps.jsonl').open('x',encoding='utf-8',newline='\n') as stream,(a.output/'gradients.jsonl').open('x',encoding='utf-8',newline='\n') as gradient_stream:
            for item in schedule:
                started=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss.reset_step_q()
                value=loss(cases[item['flat_index']]);penalty=sum(parameter.square().mean() for parameter in parameters)
                objective=value+float(cfg.get('regularization',.01))*penalty
                require(objective.requires_grad and bool(torch.isfinite(objective)),'Detached/nonfinite postcal objective')
                (objective*128).backward();norms=[];present=[];finite=[]
                for parameter in parameters:
                    exists=parameter.grad is not None;present.append(exists)
                    if exists:
                        parameter.grad.div_(128);finite.append(bool(torch.isfinite(parameter.grad).all()));norms.append(float(parameter.grad.double().norm()))
                    else:finite.append(False);norms.append(float('nan'))
                gradient_record=dict(step=item['step'],qa_id=cases[item['flat_index']]['id'],loss_before_update=float(value.detach()),
                    parameter_gradient_present=present,parameter_gradient_finite=finite,parameter_gradient_norms=norms,
                    q_hook_count=loss.step_q_count,q_elements_all_finite=loss.step_q_finite,q_squared_norm=loss.step_q_squared)
                gradient_stream.write(json.dumps(finite_json(gradient_record),allow_nan=False)+'\n');gradient_stream.flush()
                diagnostic=gradient_diagnostic(item['step'],norms,present,finite,loss.step_q_count,loss.step_q_finite,loss.step_q_squared)
                if diagnostic['accepted_finite_zero']:zero_steps.append(item['step'])
                if diagnostic['query_finite_zero']:zero_q_steps.append(item['step'])
                for family in diagnostic['parameter_zero_families']:zero_family_counts[family]+=1
                if diagnostic['accepted_finite_zero']:
                    print(json.dumps(dict(phase='accepted_finite_zero_gradient',**gradient_record,diagnostic=diagnostic),allow_nan=False),flush=True)
                optimizer.step()
                with torch.no_grad():
                    parameters[0].clamp_(-.7,.7);parameters[1].clamp_(-.7,.7);parameters[2].clamp_(-.35,.35)
                require(all(not p.requires_grad and p.grad is None for p in model.parameters()),'Model weights acquired gradients')
                torch.cuda.synchronize();entry=dict(step=item['step'],schedule_item=item,loss_before_update=float(value.detach()),
                    regularized_before_update=float(objective.detach()),parameter_gradient_norms=norms,q_gradient_calls=loss.step_q_count,
                    q_gradient_norm=loss.step_q_squared**.5,seconds=time.perf_counter()-started)
                stream.write(json.dumps(entry,allow_nan=False)+'\n');stream.flush();logs.append(entry)
                print(json.dumps(dict(phase='postcal_step',step=item['step'],id=cases[item['flat_index']]['id'],KL=entry['loss_before_update'],seconds=entry['seconds'])),flush=True)
                del value,objective
        loss.controls.update(finite_zero_gradient_steps=zero_steps,finite_zero_query_steps=zero_q_steps,finite_zero_parameter_family_counts=zero_family_counts,gradient_policy=cfg['gradient_gate_policy'])
        final=aggregate_kl(loss,cases,weights);write_new(a.output/'final_kl.json',final)
        table=loss.table().detach().float().cpu().contiguous();require(table.shape==(3,28,4) and table.numel()*table.element_size()==1344,'Actual absolute post table shape/bytes differ')
        improved=final['source_balanced_mean']<initial['source_balanced_mean']-1e-8
        factors=dict(schema='trajkv.round8.postcal-factors.v1',status='COMPLETE' if improved else 'FAILED_TRAIN_KL_NO_DECREASE',
            post_factors=table.tolist(),post_factors_sha256=tensor_sha(table),table_values_sha256=tensor_sha(table),parameter_shape=[3,28,4],parameter_count=336,encoder_constant_dtype='float32',encoder_constant_bytes=1344,
            fixed_stage_count=42,full_priority_sha256=contract['full_priority_sha256'],selected42_sha256=contract['selected42_sha256'],
            old_f0_sha256=manifest['old_f0_sha256'],base_metadata_sha256=manifest['old_f0_sha256'],encoder_book_sha256=manifest['encoder_book_sha256'],
            train_config_sha256=sha(a.config),source_sha256=before,steps=200,seed=1212,calibration_ids=[row['id'] for row in flat],
            source_balanced_initial_KL=initial['source_balanced_mean'],source_balanced_final_KL=final['source_balanced_mean'],train_KL_strictly_decreased=improved,
            forward_controls=loss.controls,labels_used=False,model_weights_frozen=True,checkpoint='fixed final step200; not select chosen',
            post_semantics='g/Kalpha*fK; mean unchanged; S*fV; m+[.5*(1-fV)+o*fV]*S; Valpha*fV; fixedJ/sign; nestedFP16',
            scope='calibration objective evidence only; no serving task-quality, timing or exact quantized-gradient claim')
        write_new(a.output/'factors.json',factors)
        require(improved,'Final complete70Q source-balanced train KL did not strictly decrease; factors preserved but not deployable')
        require(len(logs)==200 and loss.controls['packed_calls']==loss.controls['bridge_bitwise_equal']
            and loss.controls['identity_reader_calls']==loss.controls['identity_reader_equal'] and loss.controls['identity_reader_calls']>=70*28,
            'Incomplete200 steps/identity/packed-forward evidence')
        after=input_fingerprints(a.config,cfg);require(source_fingerprints()==before and after==inputs,'Source/input changed during calibration; preserve run')
        complete=dict(status='COMPLETE',steps=200,cases=70,sources=32,labels_used=False,model_weights_frozen=True,train_KL_strictly_decreased=True)
        write_new(a.output/'complete.json',complete)
        final_marker=dict(status='COMPLETE',schema='trajkv.round8.complete-postcal-train.v1',accepted_only_after_this_marker=True,
            config_sha256=sha(a.config),source_sha256=before,input_sha256=inputs,source_inputs_unchanged=True,encoder_inputs_unchanged=True,
            canonical_sha256={name:sha(a.output/name) for name in ('manifest.json','schedule.json','teacher_contexts.json','first_probe.json','steps.jsonl','gradients.jsonl','initial_kl.json','final_kl.json','factors.json','complete.json')},
            factors_sha256=sha(a.output/'factors.json'),manifest_sha256=sha(a.output/'manifest.json'),schedule_sha256=sha(a.output/'schedule.json'),steps=200,cases=70,sources=32,
            post_encoder_constant_bytes=1344,model_weights_frozen=True,labels_used=False,train_KL_strictly_decreased=True,
            actual_packed_forward=True,dense_derivative_only_surrogate=True,primary_quality_eligible=False,task_accuracy_eligible=False)
        write_new(a.output/'complete_postcal_train.json',final_marker)
        print(json.dumps(dict(status='COMPLETE',final_marker='complete_postcal_train.json',initial_KL=initial['source_balanced_mean'],final_KL=final['source_balanced_mean'],new_encoder_bytes=1344)),flush=True)
    except BaseException as error:
        path=a.output/'failure.json'
        if not path.exists():write_new(path,dict(status='FAILED_NEW_RUN_REQUIRED',exception_type=type(error).__name__,message=str(error),source_sha256=before,
            inputs_before=inputs,complete_postcal_train_written=False,task_accuracy_eligible=False))
        raise


if __name__=='__main__':main()
