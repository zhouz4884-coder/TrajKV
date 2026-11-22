"""Native-cache bridge for single-owner segmented FP16 execution."""
from contextlib import contextmanager
from types import MethodType
import torch
from p185_single_state import SingleState
from p185_segmented_kernel import segmented_attention


class PhaseState:
    def __init__(self, model, cache, visual, indices, method, audit=False,kivi=None):
        if method.startswith('packed_'):
            from p186_packed_state import PackedState
            if audit:raise ValueError('Packed audit must use the separate reference test')
            self.states=[PackedState(x.keys,x.values,visual,kivi) for x in cache.layers]
            method=method.replace('packed_','single_')
        else:
            self.states = [SingleState(x.keys,x.values,visual,reserve=512) for x in cache.layers]
        self.indices = {}
        for i, idx in indices.items():
            if method=='single_full':break
            chosen = idx[torch.isin(idx, visual)].reshape(*idx.shape[:2], -1)
            self.indices[i] = torch.searchsorted(self.states[i].visual_pos, chosen).contiguous()
        self.method = method
        self.decoding = False
        self.audit=[]
        self.counts = {'single_state':1, 'append_calls':0}
        self.modules = [layer.self_attn for layer in model.model.language_model.layers]
        if any(hasattr(module,'_p185_read') for module in self.modules):
            raise RuntimeError('Stale phase-state hook')
        for i, module in enumerate(self.modules):
            def read(q, layer=i):
                limited = self.method=='single_static' or (self.method=='single_phase' and self.decoding)
                state=self.states[layer];selected=self.indices[layer] if limited else None
                output=segmented_attention(q,state,selected)
                if audit:
                    import torch.nn.functional as F
                    blocks=list(state.blocks(state.length,selected))
                    positions=torch.cat([p[None,None].expand(q.shape[0],state.visual_k.shape[1],-1) if p.ndim==1 else p for _,_,p in blocks],-1)
                    positions,order=positions.sort(-1)
                    gather=order[...,None].expand(*order.shape,q.shape[-1])
                    k=torch.cat([k for k,_,_ in blocks],2).gather(2,gather)
                    v=torch.cat([v for _,v,_ in blocks],2).gather(2,gather)
                    groups=q.shape[1]//k.shape[1]
                    qp=torch.arange(state.length-q.shape[-2],state.length,device=q.device)
                    mask=positions.repeat_interleave(groups,1)[:,:,None,:]<=qp[None,None,:,None]
                    expected=F.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),v.repeat_interleave(groups,1),mask)
                    error=output.float()-expected.float()
                    self.audit.append({'layer':layer,'logical_length':state.length,'query_tokens':q.shape[-2],
                                       'max_abs_error':float(error.abs().max()),
                                       'relative_l2':float(error.norm()/expected.float().norm().clamp_min(1e-12))})
                return output
            module._p185_read = read
        self.original = (cache.update,cache.get_seq_length,cache.get_mask_sizes)
        bank = self
        def update(this,key,value,layer_idx,*args,**kwargs):
            state = bank.states[layer_idx]
            state.append(key,value)
            bank.counts['append_calls'] += 1
            return state.text_k[:,:,:state.text_length],state.text_v[:,:,:state.text_length]
        def length(this,layer_idx=0): return bank.states[layer_idx].length
        def mask(this,query_length,layer_idx): return bank.states[layer_idx].length+query_length,0
        cache.update=MethodType(update,cache)
        cache.get_seq_length=MethodType(length,cache)
        cache.get_mask_sizes=MethodType(mask,cache)
        cache._single_state_append=True
        # Drop cloned combined KV owners; the experiment's immutable source is
        # retained separately for controls, never described as deployment memory.
        for item in cache.layers:
            item.keys=None
            item.values=None

    def sync_prefill(self,cache): pass

    @contextmanager
    def decode(self,cache):
        self.decoding=True
        try: yield
        finally: self.decoding=False

    def commit(self,cache): pass

    def finalize(self,cache): pass

    def reserved_bytes(self):
        return sum(s.storage_bytes() for s in self.states)+sum(x.numel()*x.element_size() for x in self.indices.values())

    def close(self,cache):
        for module in self.modules: del module._p185_read
        cache.update,cache.get_seq_length,cache.get_mask_sizes=self.original
        del cache._single_state_append
