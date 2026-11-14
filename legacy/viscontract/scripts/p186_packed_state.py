"""KIVI2-g128 visual backing; FP16 text only; no persistent visual decode copy."""
import torch
from p185_single_state import SingleState
from p176_kivi_codec import check_groups, decode_last


class PackedState(SingleState):
    def __init__(self,key,value,visual,module,reserve=512):
        super().__init__(key,value,visual,reserve)
        self.group=128
        x=self.visual_k.transpose(2,3).contiguous()
        check_groups(x,self.group);check_groups(self.visual_v,self.group)
        self.k_code,self.k_scale,self.k_min=module.triton_quantize_and_pack_along_last_dim(x,self.group,2)
        self.v_code,self.v_scale,self.v_min=module.triton_quantize_and_pack_along_last_dim(self.visual_v,self.group,2)
        del self.visual_k,self.visual_v

    def storage_bytes(self):
        return sum(self.byte_breakdown().values())

    def byte_breakdown(self):
        def size(*names):
            tensors=[getattr(self,name) for name in names]
            return sum({(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes() for t in tensors}.values())
        return {'visual_codes':size('k_code','v_code'),
                'visual_metadata':size('k_scale','k_min','v_scale','v_min'),
                'positions':size('visual_pos','text_pos'),
                'text_reserved':size('text_k','text_v')}

    def blocks(self,*args,**kwargs):
        raise RuntimeError('Packed inference must read directly in kernel; use explicit test-only reference')

    def visual_reference(self,module):
        """Test-only full materialization; never called by native execution."""
        return (decode_last(module,self.k_code,self.k_scale,self.k_min,self.group,2).transpose(2,3).contiguous(),
                decode_last(module,self.v_code,self.v_scale,self.v_min,self.group,2))
