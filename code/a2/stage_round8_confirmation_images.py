"""Stage only pinned confirmation image bytes, without opening QA labels.

Existing remote objects must match; missing objects are uploaded through new
data-volume temporary names, SHA checked, then exclusively renamed. No model,
cache cleanup, overwrite, deletion or reference-answer access is performed.
"""
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import shlex
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from remote import connect, mkdirs, REMOTE_ROOT

MEMBERSHIPS = ('round8_text_validation128_membership_v1.json',
               'round8_doc_train64_source_membership_v1.json',
               'round8_infovqa_val64_source_membership_v1.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REMOTE_CHECK = r'''
import hashlib,json,os,sys
request=json.load(sys.stdin);prefix=os.path.realpath(request['root'])
out=[]
for item in request['items']:
    path=item['remote'];resolved=os.path.realpath(path)
    assert resolved.startswith(prefix+'/'), 'Resolved image escaped authorized data volume'
    if not os.path.exists(path):out.append(dict(remote=path,status='MISSING'));continue
    assert os.path.isfile(path), 'Image target is not a regular file'
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert os.path.getsize(path)==item['bytes'] and h.hexdigest()==item['sha256'], 'Existing pinned image differs'
    out.append(dict(remote=path,status='EXACT',sha256=h.hexdigest(),bytes=item['bytes']))
print(json.dumps(out))
'''


def remote_check(client, items):
    stdin, stdout, stderr = client.exec_command(shlex.join(['/usr/bin/python3', '-B', '-c', REMOTE_CHECK]), timeout=60)
    volume = REMOTE_ROOT.split('/VLAtest/', 1)[0]
    stdin.write(json.dumps(dict(root=volume, items=items)))
    stdin.channel.shutdown_write()
    result, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError('Pinned image verification failed: ' + error[-1000:])
    return json.loads(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--upload-missing', action='store_true')
    args = parser.parse_args()
    assert not args.receipt.exists(), 'Preserve prior image staging receipt'
    items, memberships = [], {}
    for name in MEMBERSHIPS:
        path = ROOT / 'audit/a2' / name
        memberships[path.relative_to(ROOT).as_posix()] = sha(path)
        pool = json.loads(path.read_text(encoding='utf-8-sig'))
        for item in pool['images_manifest']:
            local = Path(item['local']).resolve()
            remote = item.get('remote') or REMOTE_ROOT.rstrip('/') + '/' + item['relative']
            assert remote.startswith(REMOTE_ROOT.rstrip('/') + '/data/'), 'Explicit project data target required'
            assert local.is_file() and local.stat().st_size == item['bytes'] and sha(local) == item['sha256']
            items.append(dict(local=str(local), remote=remote, bytes=item['bytes'], sha256=item['sha256']))
    assert len(items) == 256 and len({item['remote'] for item in items}) == len(items)
    client = connect()
    uploaded = []
    try:
        before = remote_check(client, items)
        missing = {row['remote'] for row in before if row['status'] == 'MISSING'}
        if args.upload_missing and missing:
            sftp = client.open_sftp()
            try:
                for item in items:
                    if item['remote'] not in missing:
                        continue
                    mkdirs(sftp, posixpath.dirname(item['remote']))
                    temporary = item['remote'] + '.round8_' + uuid.uuid4().hex + '.part'
                    sftp.put(item['local'], temporary, confirm=True)
                    remote_check(client, [{**item, 'remote': temporary}])
                    # Standard SFTP rename is exclusive; existing targets are
                    # never removed or replaced by this uploader.
                    sftp.rename(temporary, item['remote'])
                    uploaded.append(item['remote'])
            finally:
                sftp.close()
        after = remote_check(client, items) if uploaded else before
        ready = all(row['status'] == 'EXACT' for row in after)
    finally:
        client.close()
    result = dict(status='ALL256_PINNED_IMAGES_EXACT' if ready else 'MISSING_IMAGES_NO_OVERWRITE',
        schema='trajkv.round8.confirmation-image-stage.v1', memberships_sha256=memberships,
        images=256, image_bytes=sum(item['bytes'] for item in items),
        preexisting_exact=256 - len(missing), newly_uploaded=len(uploaded),
        all_targets_on_project_data_disk=True, answers_opened=False, model_used=False,
        code_sha256=sha(__file__), before=before, after=after)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key:result[key] for key in ('status', 'images', 'image_bytes', 'preexisting_exact', 'newly_uploaded', 'answers_opened')}))


if __name__ == '__main__':
    main()
