"""Observe native teacher suffix queries without altering attention outputs."""
from contextlib import contextmanager
import torch


@contextmanager
def native_suffix_queries(model,visual_positions):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import attention_bridge
    assert attention_bridge._ACTIVE_CONTEXT is None,'Native query capture is a separate teacher phase'
    _,modules=attention_bridge.language_attention_modules(model)
    identities={id(m):i for i,m in enumerate(modules)}
    assert len(modules)==28
    original=ALL_ATTENTION_FUNCTIONS['sdpa']
    had_local='sdpa' in ALL_ATTENTION_FUNCTIONS._local_mapping
    previous=ALL_ATTENTION_FUNCTIONS._local_mapping.get('sdpa')
    result={'queries_by_layer':{},'scope':'native_FP16_teacher_suffix_and_generated_history',
            'attention_output_changed':False,'references_read':False}
    def dispatch(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
        index=identities.get(id(module))
        if index is not None:
            assert query.shape[0:2]==(1,28) and key.shape[0:2]==value.shape[0:2]==(1,4)
            assert query.dtype==key.dtype==value.dtype==torch.float16
            length=key.shape[-2]
            mask=torch.ones(length,dtype=torch.bool,device=key.device);mask[visual_positions]=False
            positions=torch.arange(length,device=key.device,dtype=torch.long)
            item={'q':query[...,-1:,:].detach().cpu().contiguous().clone(),
                  'exact_keys':key[:,:,mask].detach().cpu().contiguous().clone(),
                  'exact_values':value[:,:,mask].detach().cpu().contiguous().clone(),
                  'exact_positions':positions[mask].cpu(),
                  'query_positions':positions[-1:].cpu(),'original_query_batch':query.shape[-2]}
            result['queries_by_layer'].setdefault(index,[]).append(item)
        return original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,
                        is_causal=is_causal,**kwargs)
    ALL_ATTENTION_FUNCTIONS['sdpa']=dispatch
    try:yield result
    finally:
        if had_local:ALL_ATTENTION_FUNCTIONS['sdpa']=previous
        else:ALL_ATTENTION_FUNCTIONS._local_mapping.pop('sdpa',None)
