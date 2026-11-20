"""Read-only CPU inspection of actual CalibQuant Triton cache metadata."""
import hashlib
import json
from pathlib import Path
import shlex
from remote_control import ROOT, REMOTE_ROOT, MAIN_PYTHON, connect

PROBE = r'''
import collections,hashlib,importlib.util,json,re,sys
from pathlib import Path
root=Path(sys.argv[1]); cache=root/'cache/main/triton'
families=['tiled_encoded_group_vecmat_transp_kernel','tiled_encoded_group_vecmat_kernel']
rows=[]
for family in families:
 for p in cache.glob('*/'+family+'.json'):
  d=json.loads(p.read_text());st=p.stat()
  ir=p.with_suffix('.ttir')
  body=ir.read_text() if ir.exists() else ''
  rows.append({'family':family,'relative_path':str(p.relative_to(cache)),
   'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mtime_unix':st.st_mtime,
   'metadata':d,'ttir_header':body[:4000],
   'ttir_sha256':hashlib.sha256(body.encode()).hexdigest() if body else None})
spec=importlib.util.find_spec('triton');pkg=Path(spec.origin).parent
sources={}
for rel in ['runtime/autotuner.py','runtime/jit.py','runtime/cache.py']:
 p=pkg/rel;b=p.read_bytes();sources[rel]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'source':b.decode()}
print(json.dumps({'status':'READ_ONLY_CPU_EXISTING_CACHE_METADATA','GPU_initialized':False,
 'cache_root':str(cache),'family_counts':dict(collections.Counter(x['family'] for x in rows)),
 'metadata':rows,'runtime_sources':sources}))
'''


def main():
    output=Path('${EVIDENCE_ROOT}/results/diagnostics/calib_compile_static_v1')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'remote_existing_cache_metadata.json'
    assert not target.exists(),'Preserve the original observation; use a new version for another capture.'
    client=connect()
    try:
        inp,out,err=client.exec_command(shlex.join([MAIN_PYTHON,'-B','-c',PROBE,REMOTE_ROOT]),timeout=55)
        body=out.read();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
    finally:client.close()
    data=json.loads(body)
    target.write_bytes(body)
    print(json.dumps({'output':str(target),'sha256':hashlib.sha256(body).hexdigest(),
        'family_counts':data['family_counts'],'CPU_only':True,'GPU_initialized':False}))


if __name__=='__main__':main()
