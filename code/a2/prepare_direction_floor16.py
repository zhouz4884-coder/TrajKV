"""Bounded fixed-key V1/V2/V16 dose response on existing open development."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def write(path,data):
 if path.exists():raise RuntimeError('Fixed-key diagnostic already staged')
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
 name='direction_floor16_gpu0_v1';marker=ROOT/'audit/a2/direction_floor16_design_v1.json'
 if marker.exists():raise RuntimeError('Fixed-key diagnostic already frozen')
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text());lookup={v['name']:v for v in base['variants']}
 samples=json.loads((ROOT/'configs/a2/docvqa_next_dev64.json').read_text(encoding='utf-8'))[:16];sp=ROOT/'configs/a2/docvqa_dev16_floor.json';write(sp,samples)
 variants=[lookup['full8'],lookup['direction_vq2_v1'],dict(lookup['direction_vq2_v1'],name='direction_vq2_v2',bits_v=2,v_encoder='rtn'),
  dict(lookup['direction_vq2_v1'],name='direction_vq2_v16_diagnostic',bits_v=16,v_encoder='rtn')]
 config=copy.deepcopy(base);config.update(start=0,count=16,variants=variants,diagnostic_protocol='fixed_K2_vector_V_precision_dose_response')
 config['arguments'].update(samples=REMOTE_ROOT+'/configs/a2/'+sp.name,max_patches=8);cp=ROOT/'configs/a2'/f'{name}.json';write(cp,config)
 frozen=json.loads((ROOT/'audit/a2/book_confirmation_v1.json').read_text(encoding='utf-8'))
 wait={'timing_launch':REMOTE_ROOT+'/jobs/book_timing_gpu0_v1.launch.json','timing_complete':REMOTE_ROOT+'/results/book_timing_gpu0_v1/complete.json',
  'dependencies':[{'complete':REMOTE_ROOT+f'/results/{job}/complete.json','launch':REMOTE_ROOT+f'/jobs/{job}.launch.json'} for job in frozen['jobs']],
  'quality_config':REMOTE_ROOT+'/configs/a2/'+cp.name}
 wp=ROOT/'configs/a2'/f'{name}_guard.json';write(wp,wait)
 job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_guarded_followup.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
 write(ROOT/'configs/jobs'/f'{name}.json',job)
 write(marker,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'job':name,'images':16,'free_sessions':64,'variants':variants,
  'selection':'first16 existing development rows, no error filtering; only interprets remaining K/V fidelity bottleneck',
  'scope':'V16 and V2 exceed the1.8-bit deployment budget and are high-storage diagnostic controls; V2 uses valid RTN2 encoding, not the1-bit Lloyd encoder',
  'ordering':'after all four independent confirmation jobs; before existing quiet timing; no frozen quality/config changes; original timing waiter held/released with process identity checks and finally handler',
  'source_sha256':hashlib.sha256((ROOT/'code/a2/run_guarded_followup.py').read_bytes()).hexdigest()})
 print(json.dumps({'job':name,'sessions':64}))

if __name__=='__main__':main()
