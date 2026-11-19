"""Index exact archive content separately from predicted LF Git blob identity.

No Git or network operations. Root publication supplies an optional actual tree
map for final coverage verification. Raw records may remain archive-only.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import zipfile

TEXT = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.svg', '.log'}
BINARY = {'.png', '.pdf'}
PRIVATE = {'chat.md', 'cha.md', 'AGENTS.md', 'remote.py', 'check_github_access.py'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_name(name):
    path = PurePosixPath(name)
    require('maintenance_20261118' not in path.parts, 'Raw maintenance material is private')
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
            'Unsafe content path: ' + name)
    require(path.name not in PRIVATE and not re.search(r'^chat|^DS(?:[_. -]|\d|$)|credential|auth[_-]?helper|connection', path.name, re.I)
            and not (set(path.parts) & {'.git', '__pycache__', 'images', 'models', 'weights', 'private', 'chats', 'conversations'}),
            'Prohibited private/raw asset: ' + name)
    require(path.suffix.lower() in TEXT | BINARY, 'Unsupported public content suffix: ' + name)
    require(path.suffix.lower() not in BINARY or path.parts[0] == 'figures',
            'Binary image/PDF assets must be generated figures, not original dataset media: ' + name)


def identities(data, name):
    normalized = data.replace(b'\r\n', b'\n') if PurePosixPath(name).suffix.lower() in TEXT else data
    blob = b'blob ' + str(len(normalized)).encode() + b'\0' + normalized
    exact = digest(data)
    return dict(content_ref='sha256:' + exact, archive_exact_sha256=exact, archive_exact_bytes=len(data),
        git_normalized_body_sha256=digest(normalized), git_normalized_body_bytes=len(normalized),
        predicted_git_blob_oid_sha1=hashlib.sha1(blob).hexdigest(),
        git_normalization='CRLF to LF only; BOM and all other bytes retained' if normalized != data else 'already LF or binary unchanged')


def build(args):
    require(not args.output.exists(), 'Refusing to replace a frozen content index')
    source = read(args.source_index)
    entries = source['files']
    payload_files = {path.relative_to(args.payload).as_posix(): path for path in args.payload.rglob('*') if path.is_file()}
    files = {}
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive member names')
        for name in sorted(entries):
            safe_name(name)
            require(name in names, 'Source-index item absent from archive: ' + name)
            data = archive.read(name)
            expected = entries[name].get('public_sha256', entries[name].get('archive_exact_sha256'))
            require(expected == digest(data), 'Exact public archive SHA differs: ' + name)
            if name in payload_files:
                require(payload_files[name].read_bytes() == data, 'Browsable payload differs from exact archive content: ' + name)
            files[name] = dict(**identities(data, name), source_sha256=entries[name].get('source_sha256'),
                archive_entry=name, expected_in_git=name in payload_files,
                git_add_force_required=name.startswith('configs/data/') and name in payload_files,
                role=entries[name].get('role', 'derived public content'))
    require(set(payload_files) <= set(files), 'Browsable payload contains files omitted from source index')
    result = dict(schema='trajkv.round8.fixed-content-index.v1', status='PREPARED_LOCAL_ONLY',
        archive=dict(name=args.archive.name, sha256=file_digest(args.archive), bytes=args.archive.stat().st_size),
        source_index_sha256=digest(args.source_index.read_bytes()), files=files,
        git_stage_plan=dict(normal_paths=[name for name, item in files.items() if item['expected_in_git'] and not item['git_add_force_required']],
                            force_paths=[name for name, item in files.items() if item['git_add_force_required']],
                            archive_only_paths=[name for name, item in files.items() if not item['expected_in_git']]),
        hash_scope='content_ref fixes exact derived archive bytes; source_sha256 fixes original preserved bytes; LF Git bodies and Git blob OIDs are separate identities',
        git_scope='predicted SHA1 blob OIDs assume the documented text/LF attributes; root must compare an actual post-stage/post-commit tree map',
        public_write_performed=False, git_operation_performed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


def verify(args):
    result = read(args.verify_index)
    require(result['schema'] == 'trajkv.round8.fixed-content-index.v1', 'Unsupported content index schema')
    require(file_digest(args.archive) == result['archive']['sha256'], 'Whole exact archive fingerprint changed')
    tree = read(args.git_tree_map) if args.git_tree_map else None
    if tree is not None and 'files' in tree:
        tree = tree['files']
    checked = 0
    with zipfile.ZipFile(args.archive) as archive:
        for name, expected in result['files'].items():
            safe_name(name)
            actual = identities(archive.read(name), name)
            require(all(actual[key] == expected[key] for key in actual), 'Exact archive or normalized Git identity mismatch: ' + name)
            if expected['expected_in_git']:
                require((args.payload / name).is_file() and (args.payload / name).read_bytes() == archive.read(name),
                        'Expected browsable public payload missing/changed: ' + name)
                if tree is not None:
                    require(name in tree, 'Actual Git tree omitted expected path, including ignored metadata: ' + name)
                    value = tree[name]
                    oid = value.get('oid', value.get('blob_oid')) if isinstance(value, dict) else value
                    require(oid == expected['predicted_git_blob_oid_sha1'], 'Actual Git LF blob differs from predicted content: ' + name)
                    checked += 1
    return dict(status='PASS', archive_exact_bytes_verified=True, exact_vs_git_LF_identities_distinguished=True,
        files=len(result['files']), actual_git_tree_paths_checked=checked, actual_git_tree_verified=tree is not None,
        force_metadata_paths=result['git_stage_plan']['force_paths'],
        public_write_performed=False, git_operation_performed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify-index', type=Path)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--payload', type=Path, required=True)
    p.add_argument('--source-index', type=Path)
    p.add_argument('--git-tree-map', type=Path,
                   help='Root-supplied JSON {relative_path: actual_blob_oid}; this helper never invokes Git')
    args = p.parse_args()
    require(args.verify_index or args.source_index, '--source-index required when building')
    value = verify(args) if args.verify_index else build(args)
    print(json.dumps(value if args.verify_index else dict(status=value['status'], files=len(value['files']),
        force_metadata_paths=value['git_stage_plan']['force_paths'], public_write_performed=False, git_operation_performed=False)))
