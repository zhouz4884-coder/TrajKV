"""Canonical tensor storage accounting for this research cycle."""
def unique_storage_bytes(tensors):
    return sum({(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes()
                for t in tensors if t is not None}.values())

def tensor_payload_bytes(tensors):
    return sum(t.numel()*t.element_size() for t in tensors if t is not None)

def packed_pair_bits(k_bits,v_bits,group=128,base_metadata_bits=32,
                     k_residual_bits=0,v_sign=False):
    k=k_bits+base_metadata_bits/group
    v=v_bits+base_metadata_bits/group
    if k_residual_bits:k+=k_residual_bits+base_metadata_bits/group
    if v_sign:v+=1+16/group
    return {'k_bits_per_element':k,'v_bits_per_element':v,
            'bits_per_pair':k+v,'relative_to_fp16_pair':(k+v)/32}
