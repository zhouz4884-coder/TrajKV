"""CPU dtype trace of original CalibQuant AST with typed kernel-output stubs.

The stubs model only the source-declared output dtype (FP16), not CUDA
arithmetic. This audits type promotion and performs no inference or scoring.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'third_party/calibquant/patch_internvl.py'
    kernel_source=ROOT/'third_party/calibquant/calibquant.py'
    infer_source=ROOT/'third_party/calibquant/infer.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='forward_internlm2_calibquant')
    assigns=sorted([node for node in ast.walk(fn) if isinstance(node,ast.Assign)],key=lambda node:node.lineno)
    def target(node,name):return any(isinstance(item,ast.Name) and item.id==name for item in node.targets)
    def statement(name,number=0):return copy.deepcopy([node for node in assigns if target(node,name)][number])
    def execute(nodes,ns):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(source),'exec'),ns)
    visual=next(node for node in assigns if target(node,'attn_weights_quant_v') and any(isinstance(call,ast.Name) and call.id=='fused_kernel' for call in ast.walk(node.value)))
    normalize=next(node for node in ast.walk(fn) if isinstance(node,ast.If) and isinstance(node.test,ast.Attribute) and node.test.attr=='normalize')
    concat=next(node for node in assigns if target(node,'attn_weights_quant') and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='cat')
    softmax=next(node for node in assigns if target(node,'attn_weights') and any(isinstance(call,ast.Attribute) and call.attr=='softmax' for call in ast.walk(node.value)))
    first_value=next(node for node in assigns if target(node,'attn_output_v') and any(isinstance(call,ast.Name) and call.id=='fused_kernel_wv' for call in ast.walk(node.value)))
    value_statements=[node for node in assigns if first_value.lineno<=node.lineno<=first_value.lineno+11]
    dtype_runs=[]
    torch.manual_seed(20261120)
    for dtype in (torch.bfloat16,torch.float16):
        events=[];kernel_events=[]
        def qk(lut,weights,encoded):
            kernel_events.append({'kernel':'QK','lut_dtype':str(lut.dtype),'weights_dtype':str(weights.dtype),'output_dtype':'torch.float16'})
            return torch.tensor([-8.,-2.,2.,8.],dtype=torch.float16).reshape(1,1,1,4).expand(1,2,1,4).clone()
        def wv(lut,weights,encoded):
            kernel_events.append({'kernel':'WV','lut_dtype':str(lut.dtype),'weights_dtype':str(weights.dtype),'output_dtype':'torch.float16'})
            return torch.ones(1,2,1,128,dtype=torch.float16)
        self_ref=SimpleNamespace(scale_key=torch.full((1,2,1,128),.25,dtype=dtype),zero_point_key=torch.ones(1,2,1,128,dtype=dtype),
            scale_value=torch.full((1,2,1,128),.2,dtype=dtype),zero_point_value=torch.ones(1,2,1,128,dtype=dtype),
            head_dim=128,unique_values_k=torch.arange(2,dtype=dtype),unique_values_v=torch.arange(2,dtype=torch.float16),
            sprompt_len=0,image_token_num=4,normalize=True)
        q=torch.randn(1,2,1,128).to(dtype)
        ns={'torch':torch,'math':math,'nn':torch.nn,'self':self_ref,'query_states':q,
            'key_states_quant_s':torch.randn(1,2,0,128).to(dtype),'key_states_quant_t':torch.randn(1,2,2,128).to(dtype),
            'value_states':torch.randn(1,2,2,128).to(dtype),'encoded_features_k':torch.zeros(1,2,4,16,dtype=torch.uint8),
            'encoded_features_v':torch.zeros(1,2,4,16,dtype=torch.uint8),'fused_kernel':qk,'fused_kernel_wv':wv}
        nodes=[statement('attn_weights_quant_s'),statement('attn_weights_quant_t'),statement('bias_k',0),statement('bias_k',1),visual,normalize,concat,
               statement('attn_weights',0),softmax,*value_statements]
        for node in nodes:
            execute([node],ns)
            if isinstance(node,ast.Assign):
                for item in node.targets:
                    if isinstance(item,ast.Name) and isinstance(ns.get(item.id),torch.Tensor):
                        events.append({'line':node.lineno,'name':item.id,'dtype':str(ns[item.id].dtype)})
            elif node is normalize:
                for name in ('current_max','current_min','target_max','target_min','normalized_weights','attn_weights_quant_v'):
                    events.append({'line':node.lineno,'name':name,'dtype':str(ns[name].dtype)})
        dtype_runs.append({'model_input_dtype':str(dtype),'trace':events,'declared_kernel_dtype_stubs':kernel_events,
                           'calibrated_visual_logits_dtype':str(ns['attn_weights_quant_v'].dtype)})
    infer_tree=ast.parse(infer_source.read_text(encoding='utf-8'))
    def assigned(name):
        node=next(node for node in infer_tree.body if isinstance(node,ast.Assign) and target(node,name))
        return {'line':node.lineno,'expression':ast.unparse(node.value)}
    model_patch=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='patch_model')
    result={'status':'PASS_CPU_SOURCE_AST_DTYPE_TRACE','GPU_used':False,'model_weights_loaded':False,'quality_scored':False,
        'infer_model_dtype':assigned('dtype'),'infer_example_quant_method':assigned('quant_method'),
        'patch_model_quant_method_default':ast.literal_eval(model_patch.args.defaults[0]),
        'source_calib_decode_final_dtype':'hard-coded torch.bfloat16',
        'promotion_rule':{'Half_plus_BFloat16':str(torch.promote_types(torch.float16,torch.bfloat16)),
                          'Half_plus_Half':str(torch.promote_types(torch.float16,torch.float16))},
        'runs':dtype_runs,'kernel_stub_scope':'Only source-declared FP16 return dtype; does not emulate CUDA numeric values',
        'interpretation':'Default BF16 model plus hard-FP16 source kernel results promotes visual affine/calibration to FP32; all-FP16 common-model port has different intermediate precision even with FP32 bias SUM/DIV then Half cast',
        'torch_version':torch.__version__,'source_sha256':{str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                                                   for path in (Path(__file__),source,kernel_source,infer_source)}}
    destination=ROOT/'audit/calibquant_source_dtype_trace_v1.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'status':result['status'],'promotion_rule':result['promotion_rule'],
                      'visual_logit_dtypes':[run['calibrated_visual_logits_dtype'] for run in dtype_runs]}))


if __name__=='__main__':main()
