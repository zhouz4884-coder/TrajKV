"""Collect complete second moments from real GPU image-boundary sessions."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import session_boundary
from bank import install_legacy
from round5_moments_bank import Round5MomentsBank


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    arg=p.parse_args()
    c=json.loads(arg.config.read_text())
    arg.output.mkdir(parents=True,exist_ok=False)
    a=argparse.Namespace(**c['arguments'])
    for key in ('model','legacy_scripts','kivi_source','samples'):
        setattr(a,key,Path(getattr(a,key)))
    for k,v in dict(max_tokens=16,turns=1,margin=0.,max_replays=0,save_states=False,
                    read_budget_bytes=None,gain_table_data=None).items():
        if not hasattr(a,k):setattr(a,k,v)
    install_legacy(a.legacy_scripts)
    torch.manual_seed(1212)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads(a.samples.read_text())
    chosen=[]
    seen=set()
    for r in rows:
        image=r.get('source_document_id') or r.get('image_id') or r.get('source_image_id')
        if image is None:raise ValueError('Missing document/image identity')
        if image in seen:continue
        seen.add(image);chosen.append(r)
        if len(chosen)>=c.get('images',16):break
    settings=c['codec']
    manifest=dict(config=c,config_sha256=sha(arg.config),samples_sha256=sha(a.samples),
        selected_ids=[r['id'] for r in chosen], labels_used=False,
        external_inputs={k:sha(v) for k,v in settings.items() if k.endswith('_path')},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    (arg.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,
        dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
    Round5MomentsBank.settings=settings
    Round5MomentsBank.sink=[]
    session_boundary.ProgressiveBank=Round5MomentsBank
    aggregate=None
    local=[]
    with torch.inference_mode(),(arg.output/'records.jsonl').open('w') as out:
        for j,row in enumerate(chosen):
            start=time.perf_counter()
            result,_=session_boundary.run_session(model,proc,row,a,'round5_collector','base')
            item=Round5MomentsBank.sink.pop()
            if aggregate is None:
                aggregate=item['layers']
            else:
                for dst,src in zip(aggregate,item['layers']):
                    for key,value in src.items():
                        if key not in ('layer','wo_metric'):dst[key]+=value
            local.extend(dict(session_id=row['id'],image_id=row.get('image_id'),
                source_document_id=row.get('source_document_id') or row.get('image_id'),**r) for r in item['local_metrics'])
            result.pop('local_metrics',None)
            out.write(json.dumps(result)+'\n');out.flush()
            print(json.dumps(dict(done=j+1,total=len(chosen),id=row['id'],seconds=time.perf_counter()-start)),flush=True)
            torch.save({'layers':aggregate,'metadata':dict(images=j+1,
                k_centering='per_image_visual_mean',q_metric='uncentered_second_moment',
                q_count_includes_grouped_heads=True,wo_metric_scope='unweighted diagonal-head proxy; actual output errors recorded separately',
                selected_ids=manifest['selected_ids'][:j+1]),'local_metrics':local},arg.output/'moments.partial.pt')
    (arg.output/'moments.partial.pt').rename(arg.output/'moments.pt')
    (arg.output/'local_metrics.json').write_text(json.dumps(local))
    (arg.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',images=len(chosen),local_samples=len(local))))


if __name__=='__main__':main()
