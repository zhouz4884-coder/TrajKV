"""Actual packed100 validation of the promising direction-code prototypes."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 kernel='direction_kernel_gpu0_v1';env={'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'}
 j={'name':kernel,'env':env,'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/check_direction.py','--output',REMOTE_ROOT+'/results/'+kernel]}
 (ROOT/'configs/jobs'/f'{kernel}.json').write_text(json.dumps(j,indent=2))
 base=json.loads((ROOT/'configs/a2/mad100_e2e_s0_v1.json').read_text());common={'method':'direction','cap':8,'store_selector_meta':False,'gain':'norm','bits_v':2,'v_layout':'token'}
 variants=[]
 def add(name,**values):v=dict(common,name=name);v.update(values);variants.append(v)
 add('direction_sign_norm_v2_token')
 add('direction_sign_projection_v2_token',gain='projection')
 add('direction_sign_norm_v2_channel',v_layout='channel')
 for power in (.5,1.):add(f'direction_sign_norm_q{power:g}_v2_token',query_scale_path=REMOTE_ROOT+'/configs/a2/query_scales20.json',query_scale_power=power)
 book=REMOTE_ROOT+'/results/direction_books_gpu1_v1/books/gaussian_vq_g4_256.json'
 add('direction_vq4_norm_v1_lloyd',direction_book_path=book,bits_v=1,v_encoder='lloyd',v_layout='channel')
 add('direction_vq4_ls_v1_lloyd',direction_book_path=book,gain='ls',bits_v=1,v_encoder='lloyd',v_layout='channel')
 add('direction_sign_norm_v1_lloyd',bits_v=1,v_encoder='lloyd',v_layout='channel')
 for shard in (0,1):
  cfg=copy.deepcopy(base);cfg.update(start=50*shard,count=50,variants=variants);name=f'direction100_s{shard}_v1'
  (ROOT/'configs/a2'/f'{name}.json').write_text(json.dumps(cfg,indent=2))
  q={'wait_complete':REMOTE_ROOT+f'/results/{kernel}/complete.json','wait_launch':REMOTE_ROOT+f'/jobs/{kernel}.launch.json',
   'tasks':[{'name':'quality','entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+f'/configs/a2/{name}.json']}]}
  (ROOT/'configs/a2'/f'{name}_queue.json').write_text(json.dumps(q,indent=2))
  je=copy.deepcopy(env);je['CUDA_VISIBLE_DEVICES']=str(shard)
  j={'name':name,'env':je,'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_queue.py','--config',REMOTE_ROOT+f'/configs/a2/{name}_queue.json','--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(j,indent=2))
 print(json.dumps({'variants':len(variants),'images':100,'packed_free_generations':100*len(variants),'numeric_cases':72}))
if __name__=='__main__':main()
