"""Short native greedy probe and real compressed-state ROI evidence.

Queries are captured by temporary per-owner attend wrappers. The frozen SDPA
dispatch and packed answer remain unchanged. No teacher/native reference K is
used; only the actual low-bank fields are transiently decoded for localization.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from attention_bridge import attention_context
from m1_protocol import m1_generate_cached


def require(value, message):
    if not value:
        raise ValueError('ROUND12_PROBE: ' + message)


def raw_question_positions(processor, batch, raw_question, *, last_tokens=4):
    tokenizer=processor.tokenizer
    require(getattr(tokenizer,'is_fast',False),'offset-mapping fast tokenizer required')
    ids=batch.ids[0].detach().cpu().tolist()
    expanded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    start=expanded.rfind(raw_question)
    require(start >= 0,'raw question not found in decoded native prompt')
    stop=start+len(raw_question)
    encoded=tokenizer(expanded,add_special_tokens=False,return_offsets_mapping=True)
    require(encoded['input_ids']==ids,'expanded native prompt tokenization is not bit-exact')
    special=set(tokenizer.all_special_ids)
    pos=[i for i,((a,b),token) in enumerate(zip(encoded['offset_mapping'],ids))
         if a<stop and b>start and token not in special and b>a]
    require(pos and all(x>=batch.boundary for x in pos),'raw question must follow image prefix')
    selected=pos[-last_tokens:]
    return selected,dict(raw_question_character_span=[start,stop],
        raw_question_all_token_positions=pos,selected_question_token_positions=selected,
        selected_token_ids=[ids[i] for i in selected],
        selected_token_pieces=tokenizer.convert_ids_to_tokens([ids[i] for i in selected]),
        expanded_prompt_retokenization_exact=True)


def decode_low_key_frame(owner):
    """Transient K in the frozen centered WHT frame, incl. Half residuals."""
    state=owner.state;book=owner.shared['book']
    require(tuple(book.shape) == (16,4),'probe localization requires actual original G4_16 low')
    code=state.direction_code
    ids=torch.stack((code & 15,code >> 4),-1).flatten(-2)
    key=(book.float()[ids.long()].flatten(-2)*state.direction_gain.float()).half()
    lookup=state.depth_residual_lookup[0]
    shifts=torch.arange(128,device=key.device)%32
    words=torch.arange(128,device=key.device)//32
    for head in range(4):
        for depth in range(2):
            slot=int(lookup[head,depth])
            if slot<0:
                continue
            packed=state.depth_residual_code[slot]
            sign=((packed[:,words] >> shifts[None]) & 1).float()*2-1
            alpha=state.depth_residual_alpha[slot].float()[:,None]
            key[0,head]=(key[0,head].float()+sign*alpha).half()
    return key


@contextmanager
def capture_student_spatial_maps(cache, question_positions, *, last_layers=6):
    from fast_wht import wht
    selected=set(map(int,question_positions))
    maps={};counts={};originals=[]
    for index in range(len(cache.layers)-last_layers,len(cache.layers)):
        owner=cache.layers[index].visual_owner
        original=owner.attend;originals.append((owner,original))
        def wrapper(q,*,_index=index,_owner=owner,_original=original,**kwargs):
            output=_original(q,**kwargs)
            positions=kwargs['query_positions'].detach().cpu().tolist()
            choose=[i for i,p in enumerate(positions) if p in selected]
            if choose and _index not in maps:
                qs=q.index_select(-2,torch.tensor(choose,device=q.device)).contiguous()
                qr=wht(qs,half=True).float().reshape(1,4,7,len(choose),128)
                keys=decode_low_key_frame(_owner).float()
                logits=torch.einsum('bghqd,bgnd->bghqn',qr,keys)*(128**-.5)
                probabilities=logits.softmax(-1)
                maps[_index]=probabilities.mean(dim=(0,1,2,3)).detach().cpu()
                counts[_index]=len(choose)
            return output
        owner.attend=wrapper
    try:
        yield dict(maps=maps,query_counts=counts)
    finally:
        for owner,original in originals:
            owner.attend=original


@torch.no_grad()
def run_low_probe(model,processor,batch,cache,raw_question,*,max_probe_tokens=3):
    require(max_probe_tokens==3,'frozen first1/first3 probe cap is3')
    require(cache.get_seq_length()==batch.boundary,'probe must start at immutable low prefix')
    qpos,span=raw_question_positions(processor,batch,raw_question)
    events=[]
    def hook(module,args,output):
        logits=output.logits[0,-1].detach().float()
        require(bool(torch.isfinite(logits).all()),'nonfinite probe logits')
        top=torch.topk(logits,5)
        events.append(dict(top5_ids=top.indices.cpu().tolist(),top5_logits=top.values.cpu().tolist(),
                           raw_top1_top2_margin=float(top.values[0]-top.values[1])))
    handle=model.register_forward_hook(hook)
    start=time.perf_counter();tokens=None
    try:
        with capture_student_spatial_maps(cache,qpos) as capture:
            with attention_context(model,cache=cache):
                generated=m1_generate_cached(model,batch,cache,max_new_tokens=max_probe_tokens)
            require(torch.equal(generated[:,:batch.full_length],batch.ids),'probe changed native prompt IDs')
            tokens=generated[0,batch.full_length:].detach().cpu().tolist()
        torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        require(tokens and len(tokens)<=3 and len(events)==len(tokens),'probe forward/token count differs')
        require(len(capture['maps'])==6 and all(v==len(qpos) for v in capture['query_counts'].values()),
                'all last6 layers must capture exact selected raw question tokens')
        spatial=torch.stack([capture['maps'][i] for i in sorted(capture['maps'])]).mean(0)
        require(bool(torch.isfinite(spatial).all()) and abs(float(spatial.sum())-1)<1e-4,
                'conditional visual localization map invalid')
        result=dict(schema='trajkv.round12.actual_low_probe.v1',
            generated_probe_tokens=tokens,actual_probe_token_count=len(tokens),
            raw_logits_events=events,first_token_margin=events[0]['raw_top1_top2_margin'],
            first3_min_margin=min(x['raw_top1_top2_margin'] for x in events),
            eos_may_shorten_probe=True,native_generation_processors_used=True,
            probe_wall_ms=elapsed,raw_question_span=span,
            captured_layers=sorted(capture['maps']),query_counts=capture['query_counts'],
            localization='equal layer/head/question-token conditional visual attention; native post-RoPE Q and actual low reconstructed WHT K',
            native_visual_reference_K_used=False,
            spatial_weights=spatial.tolist())
    finally:
        handle.remove()
        cache.crop(batch.boundary)
    require(cache.get_seq_length()==batch.boundary,'probe rollback failed')
    result['low_probe_text_KV_rolled_back_to_prefix']=True
    result['probe_is_not_a_scored_terminal_answer']=True
    return result
