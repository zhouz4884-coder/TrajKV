"""Prepare a derived public research snapshot; never publish or touch originals."""
import argparse,ast,csv,hashlib,io,json,posixpath,re,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA_FIELDS={'answer','answers','prediction','pred_answer','gt_answers','reference','references','question','questions','content','text','token_ids'}
SENSITIVE_KEYS={'password','passwd','pwd','api_key','access_token','auth_token','github_token','hf_token','secret','username','ssh_user','hostname','host'}
IP=re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
SECRET=re.compile(r'(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,}|(?:password|passwd|api_key|access_token|密码)[\"\']?\s*[:=：]\s*[\"\']([^\"\'\r\n]{4,})[\"\'])',re.I)
PUBLIC_REMOTE='''"""Optional SSH orchestration, configured by the reproducing user."""
import hashlib,os,posixpath,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT=os.environ.get('TRAJKV_REMOTE_ROOT',str(ROOT))
PYTHON=os.environ.get('TRAJKV_REMOTE_PYTHON',sys.executable)
def connect():
 import paramiko
 client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy())
 client.connect(hostname=os.environ['TRAJKV_SSH_HOST'],port=int(os.environ.get('TRAJKV_SSH_PORT','22')),username=os.environ['TRAJKV_SSH_USER'],key_filename=os.environ.get('TRAJKV_SSH_KEY'),allow_agent=True,look_for_keys=True)
 return client
def mkdirs(sftp,path):
 try:sftp.stat(path);return
 except OSError:pass
 parent=posixpath.dirname(path)
 if parent and parent!=path:mkdirs(sftp,parent)
 try:sftp.mkdir(path)
 except OSError:sftp.stat(path)
def upload(client,local,remote):
 sftp=client.open_sftp()
 try:mkdirs(sftp,posixpath.dirname(remote));sftp.put(str(local),remote)
 finally:sftp.close()
 return {'local':str(local),'remote':remote,'sha256':hashlib.sha256(Path(local).read_bytes()).hexdigest()}
'''

def sha(data):return hashlib.sha256(data).hexdigest()

class Exporter:
 def __init__(self,destination):
  self.destination=destination;self.entries={};self.findings=[];self.roots=[]
  remote=ast.parse((ROOT/'code/remote.py').read_text(encoding='utf-8'))
  remote_root=next(ast.literal_eval(n.value) for n in remote.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE_ROOT' for t in n.targets))
  model=json.loads((ROOT/'configs/a2/book_confirm_doc_s0_v1.json').read_text())['arguments']['model']
  mappings=[(model,'/path/to/InternVL3-8B-hf'),(remote_root,'/path/to/TrajKV'),(remote_root.replace('/TrajKV','/VisContract_P14_2'),'/path/to/VisContract_P14_2'),
   (str(ROOT),'/path/to/TrajKV'),(str(ROOT.parent/'VisContract_P14_2'),'/path/to/VisContract_P14_2'),(str(ROOT.parent/'benchmark_samples'),'/path/to/benchmark_samples')]
  for original,target in mappings:
   for spelling in {original,original.replace('\\','/'),original.replace('\\','\\\\')}:self.roots.append((spelling,target))
  self.roots.sort(key=lambda pair:-len(pair[0]))
 def clean_string(self,text,protect=False):
  if protect:return text
  for original,target in self.roots:text=text.replace(original,target)
  text=re.sub(r'/media/[^/\s\"\']+/[^/\s\"\']+/', '/path/to/storage/',text)
  text=re.sub(r'/home/RESEARCHER/\s\"\']+/', '/home/RESEARCHER/',text)
  text=re.sub(r'[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s\"\']+', 'C:/Users/RESEARCHER',text)
  if not protect:text=IP.sub('example.org',text)
  return text
 def clean_object(self,value,protect=False):
  if isinstance(value,dict):
   result={}
   for key,item in value.items():
    newkey=self.clean_string(key);guard=protect or key in DATA_FIELDS
    if key.lower() in SENSITIVE_KEYS and isinstance(item,str) and not guard:result[newkey]='REDACTED'
    else:result[newkey]=self.clean_object(item,guard)
   return result
  if isinstance(value,list):return [self.clean_object(item,protect) for item in value]
  if isinstance(value,str):return self.clean_string(value,protect)
  return value
 def markdown_links(self,text,relative):
  def replace(match):
   target=match.group(1).strip('<>').replace('\\','/');prefix=ROOT.as_posix()+'/'
   if target.startswith(prefix):return ']('+posixpath.relpath(target[len(prefix):],relative.parent.as_posix())+')'
   return match.group(0)
  return re.sub(r'\]\((<?[A-Za-z]:[/\\][^\n)]*>?)\)',replace,text)
 def transform(self,source,relative):
  raw=source.read_bytes();suffix=source.suffix.lower()
  if suffix in {'.png','.pdf'}:return raw
  try:text=raw.decode('utf-8-sig')
  except UnicodeDecodeError:return raw
  if suffix=='.json':
   try:
    original=json.loads(text);clean=self.clean_object(original)
    if clean==original:return raw
    return (json.dumps(clean,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
   except json.JSONDecodeError:pass
  if suffix=='.jsonl':
   output=[];changed=False
   for line in text.splitlines(keepends=True):
    if not line.strip():output.append(line);continue
    original=json.loads(line);clean=self.clean_object(original)
    if clean==original:output.append(line)
    else:output.append(json.dumps(clean,ensure_ascii=False)+'\n');changed=True
   return ''.join(output).encode('utf-8') if changed else raw
  if suffix=='.csv':
   rows=list(csv.reader(io.StringIO(text)));changed=False
   if rows:
    protected=[name in DATA_FIELDS for name in rows[0]]
    for row in rows[1:]:
     for index,value in enumerate(row):
      cleaned=self.clean_string(value,index<len(protected) and protected[index]);changed|=cleaned!=value;row[index]=cleaned
   if not changed:return raw
   stream=io.StringIO(newline='');csv.writer(stream).writerows(rows);return stream.getvalue().encode('utf-8')
  if suffix=='.md':
   text=self.markdown_links(text,relative)
   if source.name=='START_HERE_2026-11-14.md':text=re.sub(r'^最新运行状态放在.*$', '完整研究进展见[项目时间线](PROJECT_HISTORY_2026-11-14.zh.md)。原机运维断点不属于公开研究入口。',text,flags=re.M)
   if source.name in {'THEORY_quantization_snr_2026-11-13.md','PAPER_SPINE_rank1_salience_2026-11-13.md'}:text='> 历史研究草案，含尚未验证或已被后续实验修正的假设。当前结论见项目时间线与本轮收尾报告。\n\n'+text
  text=self.clean_string(text)
  # Literal credential assignments are removed from this derived copy only.
  def mask(match):
   content=match.group(1) if match.lastindex else ''
   if content.lower() in {'none','null','redacted','your_token','your_password','placeholder'}:return match.group(0)
   if match.lastindex:
    start,end=match.span(1);start-=match.start();end-=match.start();return match.group(0)[:start]+'REDACTED'+match.group(0)[end:]
   return 'REDACTED'
  text=SECRET.sub(mask,text)
  return text.encode('utf-8')
 def add(self,source,relative=None):
  relative=relative or source.relative_to(ROOT);relative=Path(relative)
  if '__pycache__' in relative.parts or '.git' in relative.parts:return
  if source.name in {'chat.md','cha.md','AGENTS.md','check_github_access.py'} or source.name.startswith('EXECUTION_CHECKPOINT') or 'publication' in source.parts:return
  data=PUBLIC_REMOTE.encode('utf-8') if relative.as_posix()=='code/remote.py' else self.transform(source,relative)
  target=self.destination/relative;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(data)
  self.entries[relative.as_posix()]={'source_sha256':sha(source.read_bytes()),'public_sha256':sha(data),'bytes':len(data),'modified_for_export':data!=source.read_bytes()}
  if source.suffix.lower() not in {'.png','.pdf'}:
   decoded=data.decode('utf-8',errors='ignore')
   for match in SECRET.finditer(decoded):
    value=match.group(1) if match.lastindex else ''
    if value.lower() not in {'none','null','redacted','your_token','your_password','placeholder'}:self.findings.append({'file':relative.as_posix(),'line':decoded.count('\n',0,match.start())+1,'kind':'credential_pattern'})

def legacy_files():
 folder=ROOT.parent/'VisContract_P14_2/scripts';names=set()
 for source in (ROOT/'code').rglob('*.py'):
  if 'publication' in source.parts:continue
  names.update(re.findall(r'(?:from|import)\s+(p\d+_[A-Za-z0-9_]+)',source.read_text(encoding='utf-8',errors='replace')))
 selected={};queue=list(names)
 while queue:
  name=queue.pop();path=folder/(name+'.py')
  if name in selected or not path.exists():continue
  selected[name]=path;queue.extend(re.findall(r'(?:from|import)\s+(p\d+_[A-Za-z0-9_]+)',path.read_text(encoding='utf-8',errors='replace')))
 return list(selected.values())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args();destination=args.destination.resolve()
 assert destination.parent==ROOT.parent and destination.name.startswith('TrajKV_public_') and destination!=ROOT
 destination.mkdir(exist_ok=True,parents=True);sentinel=destination/'.public_export_owner.json'
 if sentinel.exists():assert json.loads(sentinel.read_text())['source_identity']==sha(str(ROOT).encode())
 else:
  assert not any(destination.iterdir());sentinel.write_text(json.dumps({'source_identity':sha(str(ROOT).encode()),'role':'derived staged public copy, originals preserved'}))
 exporter=Exporter(destination)
 for old in (destination/'docs').glob('EXECUTION_CHECKPOINT*.md'):
  assert old.resolve().is_relative_to(destination);old.unlink()
 for folder in ['code','configs','docs','figures']:
  for source in (ROOT/folder).rglob('*'):
   if source.is_file() and source.suffix.lower() in {'.py','.json','.jsonl','.csv','.md','.png','.pdf','.svg','.txt'}:exporter.add(source)
 for folder in ['results','audit']:
  for source in (ROOT/folder).rglob('*'):
   if not source.is_file() or '__pycache__' in source.parts or 'job_sources' in source.parts or 'analysis_sources' in source.parts:continue
   if source.suffix.lower() in {'.json','.csv','.md'} and source.stat().st_size<8*1024*1024:exporter.add(source)
 for source in legacy_files():exporter.add(source,Path('legacy/viscontract/scripts')/source.name)
 vendors=[(ROOT.parent/'VisContract_P14_2/third_party/KIVI/quant',Path('legacy/viscontract/third_party/KIVI/quant')),
  (ROOT/'third_party/reference_ports',Path('third_party/reference_ports')),(ROOT/'third_party/textvqa_metric_source',Path('third_party/textvqa_metric_source'))]
 for source_folder,target_folder in vendors:
  if not source_folder.exists():continue
  for source in source_folder.rglob('*'):
   if source.is_file() and (source.suffix.lower() in {'.py','.md','.txt'} or source.name.upper() in {'LICENSE','NOTICE','COPYING'}):exporter.add(source,target_folder/source.relative_to(source_folder))
 for name,source_folder,target_folder in [('KIVI',ROOT.parent/'VisContract_P14_2/third_party/KIVI',Path('legacy/viscontract/third_party/KIVI')),
   ('CalibQuant',ROOT.parent/'VisContract_P14_2/third_party/CalibQuant',Path('legacy/viscontract/third_party/CalibQuant')),
   ('KVarN',ROOT/'third_party/KVarN',Path('third_party/KVarN'))]:
  for filename in ['LICENSE','NOTICE','COPYING']:
   if (source_folder/filename).is_file():exporter.add(source_folder/filename,target_folder/filename)
  if name=='CalibQuant':exporter.add(source_folder/'calibquant.py',target_folder/'calibquant.py')
 # Export original KVarN helpers used by the pinned numerical reference port.
 for relative in ['vllm/model_executor/layers/quantization/kvarn/sinkhorn.py','vllm/v1/attention/ops/kvarn_store.py','vllm/v1/attention/ops/kvarn_decode.py']:
  source=ROOT/'third_party/KVarN'/relative
  if source.is_file():exporter.add(source,Path('third_party/KVarN')/relative)
 report={'status':'PASS' if not exporter.findings else 'REVIEW_REQUIRED','generated_at_utc':datetime.now(timezone.utc).isoformat(),'role':'staged derived public snapshot; no GitHub write performed',
  'files':exporter.entries,'file_count':len(exporter.entries),'total_bytes':sum(r['bytes'] for r in exporter.entries.values()),'credential_findings':exporter.findings,
  'excluded':'original images, model weights, authentication helpers, private chat, raw job snapshots and rawJSONL are not in this browsable tree; raw evidence archives prepared separately',
  'hash_scope':'source hashes refer to original private evidence; public hashes refer to derived path-redacted files; numerical result tables and parameter JSON preserved bytewise when unchanged'}
 (destination/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 private=ROOT/'audit/publication';private.mkdir(exist_ok=True,parents=True);(private/'snapshot_preview.json').write_text(json.dumps({key:value for key,value in report.items() if key!='files'},indent=2))
 print(json.dumps({key:value for key,value in report.items() if key not in ['files','excluded','hash_scope']}))

if __name__=='__main__':main()
