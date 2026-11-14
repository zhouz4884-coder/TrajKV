"""Build an isolated next-phase affine-calibration experiment, preserving v1."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 source=Path(__file__).with_name('calibrate_trajectory.py');dest=source.with_name('calibrate_trajectory_affine.py')
 text=source.read_text(encoding='utf-8')
 def replace(old,new):
  nonlocal text
  if text.count(old)!=1:raise RuntimeError('Unexpected template count: '+old[:70])
  text=text.replace(old,new)
 replace('only112 positive head factors are optimized against FP16 logits.',
  'positive head factors and optional center offsets are optimized against FP16 logits.')
 replace("opt=torch.optim.Adam([factors]+([value_factors] if train_value else []),lr=c.get('learning_rate',.02));records=[]",
  "train_center=c.get('train_center',False)\n key_centers=torch.nn.Parameter(torch.zeros_like(factors)) if train_center else None\n value_centers=torch.nn.Parameter(torch.zeros_like(factors)) if train_center and train_value else None\n params=[x for x in (factors,value_factors,key_centers,value_centers) if x is not None]\n opt=torch.optim.Adam(params,lr=c.get('learning_rate',.02));records=[]")
 replace("minimum=(center-.5*((1<<kbits)-1)*new_scale.float()).half()",
  "offset=key_centers[i][None,:,None,None] if train_center else 0.\n   minimum=(center+(offset-.5*((1<<kbits)-1))*new_scale.float()).half()")
 replace("vm=(r['value_center'].to('cuda')-.5*vs.float()).half()",
  "offset=value_centers[i][None,:,None,None] if train_center else 0.\n    vm=(r['value_center'].to('cuda')+(offset-.5)*vs.float()).half()")
 replace("penalty=factors.square().mean()+(value_factors.square().mean() if train_value else 0.)",
  "penalty=sum(parameter.square().mean() for parameter in params)")
 replace("(objective*128).backward();factors.grad.div_(128)\n  if train_value:value_factors.grad.div_(128)",
  "(objective*128).backward()\n  for parameter in params:parameter.grad.div_(128)")
 replace("if not finite or grad==0:raise RuntimeError('Invalid head-factor gradient')",
  "finite=finite and all(bool(torch.isfinite(parameter.grad).all()) for parameter in params)\n  if not finite or grad==0:raise RuntimeError('Invalid affine-parameter gradient')")
 replace("if train_value:value_factors.clamp_(-.7,.7)",
  "if train_value:value_factors.clamp_(-.7,.7)\n   if train_center:key_centers.clamp_(-.35,.35)\n   if value_centers is not None:value_centers.clamp_(-.35,.35)")
 replace("'parameter_count':factors.numel()*(2 if train_value else 1),'head_factors'",
  "'parameter_count':sum(parameter.numel() for parameter in params),\n  'key_center_offsets':key_centers.detach().cpu().tolist() if train_center else None,\n  'value_center_offsets':value_centers.detach().cpu().tolist() if value_centers is not None else None,\n  'train_center':train_center,'head_factors'")
 replace("'target_tokens_max':c.get('target_tokens',8),'base_rms':c.get('base_rms',True)",
  "'target_tokens_max':c.get('target_tokens',8),'base_rms':c.get('base_rms',True),'center_offset_scope':'per layer/KV head, fraction of the final metadata scale, codes unchanged'")
 compile(text,str(dest),'exec')
 if dest.exists() and dest.read_text(encoding='utf-8')!=text:raise RuntimeError('Refusing to replace an existing affine experiment')
 dest.write_text(text,encoding='utf-8')
 report={'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'generated':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
  'scope':'new offline differentiable calibration only; packed runtime extension and parity check required before serving claims'}
 (ROOT/'audit/a2/trajectory_affine_build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
