"""Optional SSH orchestration, configured by the reproducing user."""
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
