"""Input helpers for real image-write-only prefix extraction; no future Q seen."""
import copy
import torch
from anchor_bank import anchor_count

def prepare_row(row,cfg):
 row=copy.deepcopy(row);prefix=cfg.get('anchor_prompt','')
 row['questions'][0]=prefix+row['questions'][0]
 row['content']=[{'type':'image'},{'type':'text','text':row['questions'][0]}]
 return row

def prefix_inputs(features,end):
 length=features['input_ids'].shape[-1];result={}
 for key,value in features.items():
  if key in ('input_ids','attention_mask','token_type_ids','position_ids') and value.shape[-1]==length:result[key]=value[...,:end].contiguous()
  else:result[key]=value
 return result

def verify_fixed_prefix(processor,row,max_patches,features_fn,features,end,cfg):
 if not cfg.get('anchor_prompt'):return
 alt=copy.deepcopy(row);alt['questions'][0]=cfg['anchor_prompt']+'Explain the abstract concept of a placeholder.'
 alt['content']=[{'type':'image'},{'type':'text','text':alt['questions'][0]}]
 other=features_fn(processor,alt,max_patches)
 if not torch.equal(features['input_ids'][...,:end].cpu(),other['input_ids'][...,:end]):raise RuntimeError('Native anchor prefix depends on the actual future question')
