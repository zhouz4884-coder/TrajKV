"""CPU/network-only metadata check, independent of active GPU evaluations."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON
def main():
 name='docvqa_public_metadata_v1';job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'0','HF_HUB_DISABLE_XET':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1'},
  'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/stage_docvqa_validation.py','--metadata-only','--output',REMOTE_ROOT+'/results/'+name]}
 (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
if __name__=='__main__':main()
