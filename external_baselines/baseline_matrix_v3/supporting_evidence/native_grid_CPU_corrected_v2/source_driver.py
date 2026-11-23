"""CPU coordinate witness through the installed, pinned native pixel shuffle."""
import argparse,ast,hashlib,importlib,json,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 module=importlib.import_module('transformers.models.internvl.modeling_internvl')
 source=Path(module.__file__);body=source.read_text();tree=ast.parse(body)
 expected='fec9f18a83dea42dd32730555cac5a66ece67ef5f236f32967fed1f296d42bd6'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='pixel_shuffle' for f in n.body)]
 assert len(classes)==1
 x=torch.arange(32*32,dtype=torch.float32).reshape(1,32,32,1)
 y=getattr(module,classes[0]).pixel_shuffle(None,x,scale_factor=.5)
 assert tuple(y.shape)==(1,16,16,4)
 checks=[]
 for row in range(16):
  for col in range(16):
   observed=sorted(map(int,y[0,row,col].tolist()));want=sorted((2*row+dr)*32+(2*col+dc) for dr in (0,1) for dc in (0,1))
   assert observed==want,(row,col,observed,want)
   flat=row*16+col;assert torch.equal(y.reshape(1,256,4)[0,flat],y[0,row,col])
   checks.append({'token':flat,'tile_row':row,'tile_col':col,'native_patch_ids':observed})
 cfg=Path(a.model_path)/'config.json';c=json.loads(cfg.read_text());assert c['image_seq_length']==256 and c['downsample_ratio']==.5
 result={'schema':'trajkv.round11.native_grid_cpu_proof.v1','status':'PASS_PINNED_NATIVE_PIXEL_SHUFFLE_256_SPATIAL_COORDINATES',
  'source_modeling_sha256':expected,'source_class':classes[0],'model_config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
  'pixel_shuffle_shape':[1,16,16,4],'actual_CPU_patch_groups_verified':256,'flatten_is_row_major':True,
  'two_rows_per_32token_word':True,'vertical_pair_index_rule':'word_base + ((offset_mod32) xor16)',
  'patch_groups':checks,'input_witness_dtype':str(x.dtype),'input_witness_device':str(x.device),
  'native_transformer_weights_loaded':False,'new_model_calls':0,'new_GPU_calls':0,'new_task_answers':0,
  'qualification':'Witness confirms the installed native shuffle and flatten; actual model tile counts/visual-ID order remain checked on every inference batch.',
  'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'modeling_source_path':str(source)}
 a.output.parent.mkdir(parents=True,exist_ok=True);assert not a.output.exists()
 a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'coordinates':256,'new_model_calls':0}))
if __name__=='__main__':main()
