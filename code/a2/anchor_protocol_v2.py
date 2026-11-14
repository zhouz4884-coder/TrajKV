"""Only budgeted fixed-prefix states are inserted; no uncharged prompt tail."""
import copy
import torch
from anchor_bank import anchor_count
_LAYOUT={}

def with_text(row,text):
 result=copy.deepcopy(row);result['questions'][0]=text;result['content']=[{'type':'image'},{'type':'text','text':text}];return result

def common(a,b):
 limit=min(a.shape[-1],b.shape[-1]);different=(a[...,:limit]!=b[...,:limit]).nonzero()
 return int(different[0,-1]) if different.numel() else limit

def build_inputs(model,processor,row,max_patches,features_fn,cfg):
 features=features_fn(processor,row,max_patches);original=features['input_ids'];image_token=model.config.image_token_id
 visual=(original[0]==image_token).nonzero().flatten();end=int(visual[-1])+1;count=anchor_count(model,visual.numel(),cfg)
 info={'anchor_prefix_question_independent':True,'introduced_nonquestion_tokens':0,'budgeted_fixed_prefix_tokens':count}
 if not count:return features,visual,count,info
 prompt=cfg['anchor_prompt'];cache_key=(row['image_id'],max_patches,prompt)
 if cache_key not in _LAYOUT:
  sentinel='ZEBRA_PLACEHOLDER_173.';other='QUARTZ_SENTINEL_947.'
  bare=features_fn(processor,with_text(row,sentinel),max_patches)['input_ids']
  fixed=features_fn(processor,with_text(row,prompt+sentinel),max_patches)['input_ids']
  alternate=features_fn(processor,with_text(row,prompt+other),max_patches)['input_ids']
  assert torch.equal(fixed[...,:end],bare[...,:end])
  fixed_extent=common(fixed,alternate)-end
  structural=common(bare,fixed)-end
  if fixed_extent<cfg.get('anchor_max_tokens',16):raise RuntimeError('Fixed native prefix would reach the sentinel question')
  _LAYOUT[cache_key]=(bare.clone(),fixed.clone(),structural,fixed_extent)
 bare,fixed,structural,fixed_extent=_LAYOUT[cache_key]
 assert torch.equal(original[...,:end],fixed[...,:end]) and count<=fixed_extent
 structural=min(structural,max(0,common(original,bare)-end))
 if count<=structural:
  ids=original;introduced=0
 else:
  separator=processor.tokenizer.encode('\n',add_special_tokens=False)
  if len(separator)>count-structural:raise RuntimeError('Insufficient anchor budget for a fixed separator')
  prefix=torch.cat((fixed[...,:end+count-len(separator)],torch.tensor([separator],dtype=original.dtype)),1)
  ids=torch.cat((prefix,original[...,end+structural:]),1);introduced=count-structural
  assert ids.shape[-1]==original.shape[-1]+introduced and introduced<=count
 result=dict(features);result['input_ids']=ids
 for name in ('attention_mask','token_type_ids','position_ids'):
  if name not in features:continue
  source=features[name]
  if name=='attention_mask':result[name]=torch.ones(ids.shape,dtype=source.dtype)
  elif name=='token_type_ids':result[name]=source[...,:1].expand(*source.shape[:-1],ids.shape[-1]).clone()
  else:result[name]=torch.arange(ids.shape[-1],dtype=source.dtype)[None]
 info.update(introduced_nonquestion_tokens=introduced,structural_prefix_tokens=structural,
  actual_fixed_prefix_token_ids=ids[0,end:end+count].tolist(),
  uncharged_fixed_prompt_tail_tokens=0,
  native_prefix_proof='constructed from image and fixed sentinel-independent prefix, not actual question; actual suffix is original question IDs after shared structural framing')
 return result,visual,count,info

def prefix_inputs(features,end):
 length=features['input_ids'].shape[-1];result={}
 for key,value in features.items():
  if key in ('input_ids','attention_mask','token_type_ids','position_ids') and value.shape[-1]==length:result[key]=value[...,:end].contiguous()
  else:result[key]=value
 return result
