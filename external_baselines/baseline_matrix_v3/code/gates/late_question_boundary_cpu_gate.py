"""CPU control-flow/schema checks with an explicit fake model, no inference."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/gates'))
from late_question_compression_m1 import native_full_first,continue_after_native_first,continuation_config,check_greedy_scope


class FakeCache:
    def __init__(self,length):self.length=length
    def get_seq_length(self):return self.length


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
        self.generation_config=SimpleNamespace(num_beams=1,eos_token_id=[9],min_new_tokens=3,begin_suppress_tokens=[5])
    def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[-1],16))
    def generate(self,**kwargs):
        self.calls.append(kwargs)
        ids=kwargs['input_ids']
        if kwargs.get('return_dict_in_generate'):
            assert kwargs['max_new_tokens']==6
            criteria=kwargs['stopping_criteria'];assert not bool(criteria(ids,None)[0])
            sequence=torch.cat((ids,ids.new_tensor([[4]])),-1)
            assert bool(criteria(sequence,None)[0])
            self(input_ids=ids)
            logits=torch.zeros(1,16);logits[0,4]=3
            return SimpleNamespace(sequences=sequence,past_key_values=FakeCache(ids.shape[-1]),scores=(logits,),logits=(logits,))
        assert kwargs['past_key_values'].get_seq_length()==3
        assert ids.tolist()==[[1,2,3,4]] and kwargs['attention_mask'].shape[-1]==4
        assert kwargs['max_new_tokens']==5
        assert kwargs['generation_config'].min_new_tokens==2
        assert kwargs['generation_config'].begin_suppress_tokens is None
        return torch.cat((ids,ids.new_tensor([[7,9]])),-1)


def main():
    model=FakeModel();ids=torch.tensor([[1,2,3]])
    batch=SimpleNamespace(ids=ids,mask=torch.ones_like(ids),full_length=3,inputs={'input_ids':ids,'attention_mask':torch.ones_like(ids)})
    cache,first,audit=native_full_first(model,batch,6)
    assert first==4 and cache.get_seq_length()==3 and audit['native_prefill_forward_events'][0]['input_tokens']==3
    assert audit['original_total_answer_cap']==6 and audit['temporary_native_logits_released_before_compression']
    answer,control=continue_after_native_first(model,batch,cache,first,6)
    assert answer==[4,7,9] and control['continuation_new_token_cap']==5
    assert model.generation_config.min_new_tokens==3 and model.generation_config.begin_suppress_tokens==[5]
    calls=len(model.calls)
    result,meta=continue_after_native_first(model,batch,cache,9,6)
    assert result==[9] and not meta['continuation_forward_needed'] and len(model.calls)==calls
    result,meta=continue_after_native_first(model,batch,cache,4,1)
    assert result==[4] and not meta['continuation_forward_needed'] and len(model.calls)==calls
    refused=False
    try:continue_after_native_first(model,batch,FakeCache(4),4,6)
    except RuntimeError:refused=True
    assert refused
    unsupported=SimpleNamespace(num_beams=1,stop_strings=['test'])
    refused=False
    try:check_greedy_scope(unsupported)
    except ValueError:refused=True
    assert refused
    result={'status':'PASS_CPU_BOUNDARY_CONTROL_FLOW_SCHEMA','GPU_used':False,'real_model_inference':False,
        'fake_outputs_are_benchmark_results':False,
        'checks':['native_y1_uses_original_total_cap_not_max_new1','cache_prompt_excludes_unprocessed_y1',
                  'continuation_keeps_complete_P_plus_y1_history','y1_preserved_once_and_total_cap_respected',
                  'relative_min_new_count_and_begin_suppression_adjusted','native_generation_config_not_mutated',
                  'EOS_native_y1_stops_without_compressed_read','cap1_stops_without_compressed_read',
                  'already_processed_y1_cache_rejected','unsupported_stop_strings_not_silently_changed'],
        'source_sha256':{str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                           (Path(__file__),ROOT/'code/gates/late_question_compression_m1.py')},
        'target_runtime_source_audit':'audit/calib_execution_boundary_source_v1.json'}
    with (ROOT/'audit/late_question_boundary_cpu_gate_v1.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),'GPU_used':False}))


if __name__=='__main__':main()
