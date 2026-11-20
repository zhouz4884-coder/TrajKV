import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);jobfile=Path(sys.argv[2]);j=json.loads(jobfile.read_text());name=j['name']
assert name.replace('_','').replace('-','').isalnum()
receipt=root/'jobs'/(name+'.launch.json');snapshot=root/'jobs'/(name+'.source')
if receipt.exists():
    old=json.loads(receipt.read_text());assert old['job_sha256']==hashlib.sha256(jobfile.read_bytes()).hexdigest()
    observed=dict(old);observed['status']='ALREADY_LAUNCHED_INSPECT_EXISTING'
    print(json.dumps(observed));raise SystemExit(0)
assert not snapshot.exists(), 'Incomplete preparation preserved; use a new job name'
volume=Path(sys.argv[3]).resolve();assert str(root.resolve()).startswith(str(volume)+'/')
env=dict(os.environ);env.update(j.get('env',{}))
cache={}
for key in ('TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH',
 'PYTHONPYCACHEPREFIX','HF_HOME','HF_HUB_CACHE','TORCH_HOME','TORCH_EXTENSIONS_DIR','PIP_CACHE_DIR','UV_CACHE_DIR','XDG_CACHE_HOME'):
    path=Path(env[key]).resolve();assert str(path).startswith(str(volume)+'/'), key
    path.mkdir(parents=True,exist_ok=True);cache[key]=str(path)
snapshot.mkdir();hashes={}
for folder in ('code','third_party','configs/methods','audit/upstream'):
    source_root=root/folder
    if not source_root.exists():continue
    for source in sorted(source_root.rglob('*')):
        if not source.is_file() or set(source.parts)&{'.git','__pycache__','.venv','weights','models'}:continue
        if source.suffix.lower() not in ('.py','.json','.toml','.yaml','.yml','.cu','.cpp','.h','.cuh','.npy','.txt','.md','.sh'):continue
        assert source.stat().st_size<=32*1024**2
        raw=source.read_bytes();relative=source.relative_to(root);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest()
        hashes[str(relative)]=hashlib.sha256(raw).hexdigest()
(snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2))
argv=list(j['argv'])
for index,value in enumerate(argv):
    if isinstance(value,str) and value.startswith(str(root)+'/code/') and value.endswith('.py'):
        argv[index]=str(snapshot/Path(value).relative_to(root))
env['PYTHONPATH']=str(snapshot/'code')+os.pathsep+env.get('PYTHONPATH','')
env['BASELINE_DATA_ROOT']=str(root);env['BASELINE_SOURCE_ROOT']=str(snapshot)
log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
with log.open('ab',buffering=0) as output,open(os.devnull,'rb') as null:
    worker=subprocess.Popen(argv,cwd=str(snapshot),env=env,stdin=null,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
record=dict(status='LAUNCHED',pid=worker.pid,name=name,argv_executed=argv,source_snapshot=str(snapshot),
 source_inventory_sha256=hashlib.sha256((snapshot/'sha256.json').read_bytes()).hexdigest(),
 job_sha256=hashlib.sha256(jobfile.read_bytes()).hexdigest(),log=str(log),cache_environment=cache,
 CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'),runtime_python=argv[0])
with receipt.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
