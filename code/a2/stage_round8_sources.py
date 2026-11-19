"""Upload model-side source into a separate data-disk working tree.

No job/model is launched and no global remote source or other project is edited.
The existing data-disk launcher snapshots this tree independently for each job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from remote import connect, mkdirs, REMOTE_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    if not args.version.replace('_', '').isalnum() or args.receipt.exists():
        raise ValueError('Use a new simple version/receipt')
    destination = REMOTE_ROOT + '/stages/round8_20261118_v1'
    paths = []
    for folder in ('code/a2', 'code/experiments', 'code/analysis', 'code/policies'):
        for path in (ROOT / folder).rglob('*.py'):
            if '__pycache__' in path.parts or 'tests' in path.parts:
                continue
            if re.search(r'credential|auth[_-]?helper|connection', path.name, re.I):
                continue
            paths.append(path)
    for value in ('code/bytes.py',):
        path = ROOT / value
        if path.is_file():
            paths.append(path)
    ports = ROOT / 'third_party/reference_ports'
    paths += list(ports.glob('*.py'))
    for value in ('vllm/model_executor/layers/quantization/kvarn/sinkhorn.py',
                  'vllm/v1/attention/ops/kvarn_store.py',
                  'vllm/v1/attention/ops/kvarn_decode.py'):
        path = ROOT / 'third_party/KVarN' / value
        if path.is_file():
            paths.append(path)
    client = connect()
    mapping = {}
    try:
        sftp = client.open_sftp()
        made = set()
        for path in sorted(set(paths)):
            relative = path.relative_to(ROOT).as_posix()
            remote = destination + '/' + relative
            parent = posixpath.dirname(remote)
            if parent not in made:
                mkdirs(sftp, parent); made.add(parent)
            sftp.put(str(path), remote)
            mapping[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = dict(schema='trajkv.round8.source-stage.v1',
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            version=args.version, source_root=destination, files=mapping,
            global_remote_code_modified=False, model_or_GPU_started=False,
            scope='isolated working sources; each actual launch takes its own immutable snapshot')
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2)
        remote_receipt = destination + '/source_stage_' + args.version + '.json'
        sftp.put(str(args.receipt), remote_receipt)
        sftp.close()
        print(json.dumps(dict(source_root=destination, files=len(mapping),
            receipt=str(args.receipt), GPU_started=False)))
    finally:
        client.close()


if __name__ == '__main__':
    main()
