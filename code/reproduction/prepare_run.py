"""Restore pinned benchmark image bytes and render a new local GPU run config."""
import argparse,copy,hashlib,io,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--variants',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=json.loads(a.config.read_text(encoding='utf-8'))
 def resolve(value):
  if isinstance(value,dict):return {key:resolve(item) for key,item in value.items()}
  if isinstance(value,list):return [resolve(item) for item in value]
  if isinstance(value,str):
   value=value.replace('/path/to/VisContract_P14_2',str(ROOT/'legacy/viscontract')).replace('/path/to/TrajKV',str(ROOT))
  return value
 c=resolve(c);samples_path=Path(c['arguments']['samples']);samples=json.loads(samples_path.read_text(encoding='utf-8'));variants=[v for v in c['variants'] if v['name'] in a.variants]
 assert set(a.variants)=={v['name'] for v in variants}
 dataset=samples[0]['dataset'];is_doc=dataset.startswith('docvqa');is_text=dataset=='textvqa'
 if not (is_doc or is_text):raise ValueError('This helper restores pinned DocVQA/TextVQA cohorts; legacy cohort assets have separate provenance')
 repo='lmms-lab-encoder/DocVQA' if is_doc else 'lmms-lab-encoder/textvqa';revision='539088ef8a8ada01ac8e2e6d4e372586748a265e' if is_doc else '9c0699cd19768ac5ab97568f6b3cbac4c0062884';prefix='DocVQA/validation-' if is_doc else 'data/validation-'
 os.environ.update(HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1')
 from huggingface_hub import HfApi,hf_hub_download
 import pyarrow.parquet as pq
 from PIL import Image
 info=HfApi(token=False).dataset_info(repo,revision=revision,files_metadata=True);paths=[entry.rfilename for entry in info.siblings if entry.rfilename.startswith(prefix) and entry.rfilename.endswith('.parquet')]
 wanted={h for row in samples for h in row['hashes']};found={};images=ROOT/'data/reproduced_images';images.mkdir(exist_ok=True,parents=True)
 for filename in sorted(paths):
  shard=hf_hub_download(repo,filename,repo_type='dataset',revision=revision,token=False)
  for batch in pq.ParquetFile(shard).iter_batches(batch_size=64,columns=['image']):
   for row in batch.to_pylist():
    raw=row['image']['bytes'];digest=hashlib.sha256(raw).hexdigest()
    if digest not in wanted or digest in found:continue
    with Image.open(io.BytesIO(raw)) as image:fmt=image.format;image.verify()
    extension={'JPEG':'.jpg','PNG':'.png','TIFF':'.tif','MPO':'.mpo','WEBP':'.webp'}.get(fmt,'.img');path=images/(digest+extension)
    path.write_bytes(raw);assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;found[digest]=str(path.resolve())
  print(json.dumps({'restored_images':len(found),'wanted':len(wanted)}),flush=True)
  if set(found)==wanted:break
 assert set(found)==wanted,'Some original image hashes were not found in the pinned export'
 for row in samples:row['images']=[found[h] for h in row['hashes']];row['remote_images']=list(row['images'])
 a.output.mkdir(exist_ok=False,parents=True);sp=a.output/'samples.json';sp.write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8')
 c.update(start=0,count=len(samples),variants=variants);c['arguments'].update(samples=str(sp.resolve()),model=str(a.model.resolve()),legacy_scripts=str(ROOT/'legacy/viscontract/scripts'),kivi_source=str(ROOT/'legacy/viscontract/third_party/KIVI/quant/new_pack.py'))
 cp=a.output/'config.json';cp.write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'new_config':str(cp.resolve()),'images':len(wanted),'sessions':len(samples)*len(variants),'original_files_modified':False}))

if __name__=='__main__':main()
