"""Atomically copy research outputs and source snapshots from our worker."""
import argparse,hashlib,json,posixpath
from pathlib import Path
from remote import connect,REMOTE_ROOT

def get_file(sftp,remote,local):
    attr=sftp.stat(remote)  # Check before opening any local file.
    if not attr.st_size:return False
    local.parent.mkdir(parents=True,exist_ok=True)
    part=local.with_name(local.name+'.download-part')
    sftp.get(remote,str(part))
    if part.stat().st_size!=attr.st_size:
        # A growing log is a useful snapshot, but never a completed receipt.
        if local.name=='complete.json':part.unlink();return False
    if local.name=='complete.json':
        try:json.loads(part.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):part.unlink();return False
    part.replace(local)
    return True

def get_tree(sftp,remote,local,extensions=('.py','.json','.jsonl','.csv','.md','.txt','.log')):
    import stat
    copied=0
    for item in sftp.listdir_attr(remote):
        if item.filename in ('.git','__pycache__'):continue
        source=posixpath.join(remote,item.filename);target=local/item.filename
        if stat.S_ISDIR(item.st_mode):copied+=get_tree(sftp,source,target,extensions)
        elif item.filename.endswith(extensions):copied+=int(get_file(sftp,source,target))
    return copied

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--sources',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];c=connect();s=c.open_sftp()
    try:
      for name in a.runs:
        if not name.replace('_','').replace('-','').isalnum():raise ValueError('Simple run names required')
        remote=REMOTE_ROOT+'/results/'+name;local=root/'results'/name
        local.mkdir(parents=True,exist_ok=True)
        # Remove only a zero-byte placeholder created by old non-atomic pulls.
        empty=local/'complete.json'
        if empty.exists() and empty.stat().st_size==0:
            if not empty.resolve().is_relative_to(root.resolve()):raise ValueError('Outside workspace')
            empty.unlink()
        count=get_tree(s,remote,local) if a.sources else 0
        if not a.sources:
            for entry in s.listdir(remote):
                if entry.endswith(('.json','.jsonl','.csv')):count+=int(get_file(s,remote+'/'+entry,local/entry))
        if a.sources:
            for suffix in ('.source','.launch.json','.json'):
                path=REMOTE_ROOT+'/jobs/'+name+suffix;target=root/'audit/job_sources'/(name+suffix)
                try:
                    if suffix=='.source':count+=get_tree(s,path,target)
                    else:count+=int(get_file(s,path,target))
                except OSError:pass
        done=(local/'complete.json').exists() and bool(json.loads((local/'complete.json').read_text()))
        record=local/'records.jsonl'
        n=0
        if record.exists():
            for line in record.read_text(encoding='utf-8').splitlines():
                try:json.loads(line);n+=1
                except ValueError:break
        print(json.dumps({'run':name,'complete_receipt_valid':done,'records':n,'copied':count}),flush=True)
    finally:s.close();c.close()

if __name__=='__main__':main()
