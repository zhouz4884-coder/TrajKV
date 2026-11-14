"""Reusable canonical-order decode banks and current-query selection controls."""
from contextlib import contextmanager, nullcontext
from types import MethodType
import math
import torch
import torch.nn.functional as F
from p180_multiturn_core import causal_bias, grouped_scores

PROFILE = False


def scope(name):
    return torch.profiler.record_function('P184.'+name) if PROFILE else nullcontext()


def current_scores(query, key, scaling, window=64):
    count=min(window, query.shape[-2]);start=key.shape[-2]-count
    q=query[:,:,-count:].float()
    k=key.float().repeat_interleave(query.shape[1]//key.shape[1],1)
    logits=q@k.transpose(-1,-2)*scaling
    future=torch.arange(key.shape[-2],device=key.device)[None]>torch.arange(start,key.shape[-2],device=key.device)[:,None]
    mass=logits.masked_fill(future,float('-inf')).softmax(-1).sum(2)
    pooled=torch.zeros_like(mass)
    if start:pooled[...,:start]=F.avg_pool1d(mass[...,:start],5,1,2)
    return grouped_scores(pooled,key.shape[1])


def select_current(stats, visual, cache, ratio=.25):
    result={}
    for layer,item in enumerate(cache.layers):
        scores=stats[layer].index_select(-1,visual)
        count=max(1,math.ceil(ratio*visual.numel()))
        choice=torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:count]
        chosen=visual[choice]
        text_mask=torch.ones(item.keys.shape[-2],device=visual.device,dtype=torch.bool)
        text_mask[visual]=False
        text=torch.nonzero(text_mask).flatten()[None,None].expand(item.keys.shape[0],item.keys.shape[1],-1)
        result[layer]=torch.cat((chosen,text),-1).sort(-1).values
    return result


def frozen_indices(stats,visual,cache,ratio=.25):
    """P183's exact T1 support, without materializing a discarded compact KV copy."""
    result={}
    for layer,item in enumerate(cache.layers):
        score=stats[layer]['snapkv'].index_select(-1,visual).clone()
        recent=visual>=stats[layer]['recent_start']
        count=max(1,math.ceil(ratio*visual.numel()),int(recent.sum()))
        score[...,recent]=float('inf')
        choice=torch.argsort(score,dim=-1,descending=True,stable=True)[...,:count]
        text_mask=torch.ones(item.keys.shape[-2],device=visual.device,dtype=torch.bool);text_mask[visual]=False
        text=torch.nonzero(text_mask).flatten()[None,None].expand(item.keys.shape[0],item.keys.shape[1],-1)
        result[layer]=torch.cat((visual[choice],text),-1).sort(-1).values
    return result


@contextmanager
def attention(model, indices=None, initial_length=None, capture=None, counters=None):
    """Full SDPA or the frozen P183 per-token gather, with optional charged observation."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    modules={id(layer.self_attn) for layer in model.model.language_model.layers}
    def apply(module,q,k,v,mask,dropout=0.,scaling=None,**kwargs):
        if id(module) not in modules:
            return original(module,q,k,v,mask,dropout=dropout,scaling=scaling,**kwargs)
        if hasattr(module,'_p185_read'):
            if capture is not None or dropout:
                raise RuntimeError('P185 does not support observation or dropout')
            if scaling is not None and abs(scaling-q.shape[-1]**-.5)>1e-8:
                raise RuntimeError('P185 requires standard attention scale')
            return module._p185_read(q).transpose(1,2).contiguous(),None
        kwargs['is_causal']=False
        used_k,used_v=k,v
        bias=None if q.shape[-2]==1 else causal_bias(q,k)
        if indices is not None:
            with scope('gather_indices'):
                tail=torch.arange(initial_length,k.shape[-2],device=k.device)[None,None].expand(k.shape[0],k.shape[1],-1)
                selected=torch.cat((indices[int(module.layer_idx)],tail),-1)
                gather=selected[...,None].expand(*selected.shape,k.shape[-1])
            with scope('kv_gather'):
                used_k=k.gather(2,gather);used_v=v.gather(2,gather)
                if counters is not None:
                    counters['gather_calls']=counters.get('gather_calls',0)+2
                    counters['gather_payload_bytes']=counters.get('gather_payload_bytes',0)+2*used_k.numel()*used_k.element_size()
            with scope('gather_causal_layout'):
                if q.shape[-2]>1:
                    pos=torch.arange(k.shape[-2]-q.shape[-2],k.shape[-2],device=k.device)
                    future=(selected[:,:,None,:]>pos[None,None,:,None]).repeat_interleave(q.shape[1]//k.shape[1],1)
                    bias=torch.zeros(future.shape,device=q.device,dtype=q.dtype).masked_fill_(future,float('-inf'))
        with scope('attention'):
            output=original(module,q,used_k,used_v,bias,dropout=dropout,scaling=scaling,**kwargs)
        if capture is not None:
            with scope('current_observation'):
                capture[int(module.layer_idx)]=current_scores(q,k,scaling or q.shape[-1]**-.5)
        return output
    ALL_ATTENTION_FUNCTIONS.register('sdpa',apply)
    try:yield
    finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


@contextmanager
def instrument_updates(cache, counters):
    original=cache.update
    def update(self,key,value,layer_idx,*args,**kwargs):
        with scope('dense_cache_update'):
            k,v=original(key,value,layer_idx,*args,**kwargs)
        counters['dense_update_calls']=counters.get('dense_update_calls',0)+1
        written=(key,value) if getattr(cache,'_single_state_append',False) else (k,v)
        counters['dense_update_write_bytes']=counters.get('dense_update_write_bytes',0)+sum(x.numel()*x.element_size() for x in written)
        return k,v
    cache.update=MethodType(update,cache)
    try:yield
    finally:cache.update=original


class WorkingSet:
    def __init__(self, cache, indices, reserve=512):
        self.keys=[];self.values=[];self.lengths=[];self.logical=[];self.ids={};self.ids_logical={}
        self.initial_lengths=[];self.initial_logical=[]
        self.reserve=reserve
        self.counts={'initial_gathers':0,'refresh_gathers':0,'fresh_rows':0,'moved_rows':0,
                     'unchanged_rows':0,'append_rows':0,'growths':0,'commit_calls':0,'commit_write_bytes':0,
                     'block_refresh_rows':0}
        with scope('bank_initialization'):
            for i,item in enumerate(cache.layers):
                selected=indices[i].long();n=selected.shape[-1]
                capacity=n+reserve
                self.keys.append(item.keys.new_empty((*item.keys.shape[:2],capacity,item.keys.shape[-1])))
                self.values.append(item.values.new_empty((*item.values.shape[:2],capacity,item.values.shape[-1])))
                gather=selected[...,None].expand(*selected.shape,item.keys.shape[-1])
                torch.gather(item.keys,2,gather,out=self.keys[i][:,:,:n])
                torch.gather(item.values,2,gather,out=self.values[i][:,:,:n])
                self.lengths.append(n);self.logical.append(item.keys.shape[-2])
                self.initial_lengths.append(n);self.initial_logical.append(item.keys.shape[-2])
                self.ids[i]=selected;self.ids_logical[i]=item.keys.shape[-2]
                self.counts['initial_gathers']+=2

    def ensure(self,i,needed):
        if needed<=self.keys[i].shape[-2]:return
        capacity=max(needed+self.reserve,self.keys[i].shape[-2]*2)
        for buffers in (self.keys,self.values):
            old=buffers[i];new=old.new_empty((*old.shape[:2],capacity,old.shape[-1]))
            new[:,:,:self.lengths[i]].copy_(old[:,:,:self.lengths[i]])
            buffers[i]=new
        self.counts['growths']+=1

    def append(self,i,key,value):
        n=key.shape[-2];start=self.lengths[i]
        self.ensure(i,start+n)
        self.keys[i][:,:,start:start+n].copy_(key)
        self.values[i][:,:,start:start+n].copy_(value)
        self.lengths[i]+=n;self.logical[i]+=n
        self.counts['append_rows']+=key.shape[0]*key.shape[1]*n

    def sync_prefill(self,cache):
        with scope('bank_append_prefill'):
            for i,item in enumerate(cache.layers):
                start=self.logical[i];end=item.keys.shape[-2]
                if end<start:raise RuntimeError('Backing history shortened')
                if end>start:self.append(i,item.keys[:,:,start:end],item.values[:,:,start:end])

    def current_ids(self,i):
        tail=torch.arange(self.ids_logical[i],self.logical[i],device=self.ids[i].device)[None,None].expand(*self.ids[i].shape[:2],-1)
        return torch.cat((self.ids[i],tail),-1)

    def refresh(self,cache,indices,incremental=False):
        label='bank_refresh_block' if incremental=='block' else ('bank_refresh_incremental' if incremental else 'bank_refresh_rebuild')
        with scope(label):
            for i,item in enumerate(cache.layers):
                new=indices[i];n=new.shape[-1]
                if n!=self.lengths[i]:raise RuntimeError('Changed support budget')
                if incremental=='block':
                    prefix=self.initial_lengths[i]
                    if n-prefix!=self.logical[i]-self.initial_logical[i]:raise RuntimeError('Text tail layout changed')
                    changed=new[:,:,:prefix]
                    gather=changed[...,None].expand(*changed.shape,item.keys.shape[-1])
                    torch.gather(item.keys,2,gather,out=self.keys[i][:,:,:prefix])
                    torch.gather(item.values,2,gather,out=self.values[i][:,:,:prefix])
                    self.counts['block_refresh_rows']+=changed.numel()
                elif not incremental:
                    gather=new[...,None].expand(*new.shape,item.keys.shape[-1])
                    torch.gather(item.keys,2,gather,out=self.keys[i][:,:,:n])
                    torch.gather(item.values,2,gather,out=self.values[i][:,:,:n])
                    self.counts['refresh_gathers']+=2
                else:
                    old=self.current_ids(i)
                    if old.shape!=new.shape:raise RuntimeError('Changed support budget')
                    rank=torch.searchsorted(old.contiguous(),new.contiguous()).clamp(max=n-1)
                    common=old.gather(-1,rank)==new
                    position=torch.arange(n,device=new.device)[None,None].expand_as(new)
                    moved=common & (rank!=position);fresh=~common
                    heads=torch.arange(new.shape[0]*new.shape[1],device=new.device).reshape(*new.shape[:2],1).expand_as(new)
                    source_moved=(heads*self.keys[i].shape[-2]+rank)[moved]
                    target_moved=(heads*self.keys[i].shape[-2]+position)[moved]
                    source_fresh=(heads*item.keys.shape[-2]+new)[fresh]
                    target_fresh=(heads*self.keys[i].shape[-2]+position)[fresh]
                    for bank,full in ((self.keys[i],item.keys),(self.values[i],item.values)):
                        flat=bank.view(-1,bank.shape[-1])
                        moving=flat.index_select(0,source_moved)
                        incoming=full.reshape(-1,full.shape[-1]).index_select(0,source_fresh)
                        flat.index_copy_(0,target_moved,moving)
                        flat.index_copy_(0,target_fresh,incoming)
                    self.counts['fresh_rows']+=source_fresh.numel()
                    self.counts['moved_rows']+=source_moved.numel()
                    self.counts['unchanged_rows']+=new.numel()-source_fresh.numel()-source_moved.numel()
                self.ids[i]=new;self.ids_logical[i]=self.logical[i]

    @contextmanager
    def decode(self,cache):
        if cache.offloading:raise RuntimeError('Offloading not supported by this prototype')
        starts=list(self.lengths)
        original_update=cache.update;original_length=cache.get_seq_length;original_mask=cache.get_mask_sizes
        bank=self
        def update(this,key,value,layer_idx,*args,**kwargs):
            if key.shape[-2]!=1:raise RuntimeError('Compact decode requires one new token')
            with scope('compact_append'):
                bank.append(layer_idx,key,value)
            n=bank.lengths[layer_idx]
            return bank.keys[layer_idx][:,:,:n],bank.values[layer_idx][:,:,:n]
        def length(this,layer_idx=0):return bank.logical[layer_idx]
        def mask_size(this,query_length,layer_idx):return bank.logical[layer_idx]+query_length,0
        cache.update=MethodType(update,cache);cache.get_seq_length=MethodType(length,cache);cache.get_mask_sizes=MethodType(mask_size,cache)
        try:yield
        finally:
            cache.update=original_update;cache.get_seq_length=original_length;cache.get_mask_sizes=original_mask
            self.pending_starts=starts

    def commit(self,cache):
        with scope('backing_commit'):
            for i,item in enumerate(cache.layers):
                start=self.pending_starts[i];end=self.lengths[i]
                if end>start:
                    item.keys=torch.cat((item.keys,self.keys[i][:,:,start:end]),-2)
                    item.values=torch.cat((item.values,self.values[i][:,:,start:end]),-2)
                    self.counts['commit_calls']+=1
                    self.counts['commit_write_bytes']+=item.keys.numel()*item.keys.element_size()+item.values.numel()*item.values.element_size()
                if item.keys.shape[-2]!=self.logical[i]:raise RuntimeError('Backing commit length mismatch')

    def reserved_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.keys+self.values)


class DeferredWorkingSet(WorkingSet):
    """Merge a generated tail with the next prefill instead of copying full backing twice."""
    def __init__(self,cache,indices,reserve=512):
        super().__init__(cache,indices,reserve)
        if cache.offloading:raise RuntimeError('Offloading is not supported')
        self.pending=[0]*len(cache.layers);self.pending_start=list(self.lengths)
        self.phase='prefill'
        self.original_methods=(cache.update,cache.get_seq_length,cache.get_mask_sizes)
        bank=self
        def update(this,key,value,layer_idx,*args,**kwargs):
            i=layer_idx
            if bank.phase=='decode':
                if key.shape[-2]!=1:raise RuntimeError('Decode append must contain one token')
                if bank.pending[i]==0:bank.pending_start[i]=bank.lengths[i]
                with scope('compact_append'):bank.append(i,key,value)
                bank.pending[i]+=1
                return bank.keys[i][:,:,:bank.lengths[i]],bank.values[i][:,:,:bank.lengths[i]]
            item=this.layers[i]
            with scope('prefill_backing_merge'):
                if bank.pending[i]:
                    start=bank.pending_start[i];end=start+bank.pending[i]
                    item.keys=torch.cat((item.keys,bank.keys[i][:,:,start:end],key),-2)
                    item.values=torch.cat((item.values,bank.values[i][:,:,start:end],value),-2)
                    bank.pending[i]=0
                else:
                    item.keys=torch.cat((item.keys,key),-2)
                    item.values=torch.cat((item.values,value),-2)
            return item.keys,item.values
        def length(this,layer_idx=0):return this.layers[layer_idx].keys.shape[-2]+bank.pending[layer_idx]
        def mask_size(this,query_length,layer_idx):return length(this,layer_idx)+query_length,0
        cache.update=MethodType(update,cache);cache.get_seq_length=MethodType(length,cache);cache.get_mask_sizes=MethodType(mask_size,cache)

    @contextmanager
    def decode(self,cache):
        self.phase='decode'
        try:yield
        finally:self.phase='prefill'

    def commit(self,cache):
        # Pending states are already in the bank; the next prefill merges them once.
        pass

    def dense_layer(self,cache,i):
        item=cache.layers[i]
        if not self.pending[i]:return item.keys,item.values
        start=self.pending_start[i];end=start+self.pending[i]
        return (torch.cat((item.keys,self.keys[i][:,:,start:end]),-2),
                torch.cat((item.values,self.values[i][:,:,start:end]),-2))

    def finalize(self,cache):
        with scope('final_backing_commit'):
            for i,item in enumerate(cache.layers):
                if self.pending[i]:
                    item.keys,item.values=self.dense_layer(cache,i)
                    self.counts['commit_calls']+=1
                    self.counts['commit_write_bytes']+=item.keys.numel()*item.keys.element_size()+item.values.numel()*item.values.element_size()
                    self.pending[i]=0
                if item.keys.shape[-2]!=self.logical[i]:raise RuntimeError('Final backing length mismatch')
        cache.update,cache.get_seq_length,cache.get_mask_sizes=self.original_methods
