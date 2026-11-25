"""Root-pinned reuse of a real R13 export that failed only at semantics.

This is not R12 PinnedReuse and does not confer publication eligibility. It
only copies fingerprint-verified unchanged bodies/JSONL identities; the new
export must still run the full fresh public cold verifier.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from round12_public_core_v1 import NAMESPACE, canonical, require, safe_relative, sha

SCHEMA='trajkv.round13.root_pinned_failed_export_reuse.v1'
OBJECT_NAMES=('sealed_scope','index','archive','audit','failure_trace')


def _read_pinned(item):
    path=Path(item['path']).resolve()
    require(path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],
            'R13 cache Root-pinned object differs: '+path.name)
    return path


def _verify_jsonl_lines(path,entry,side_path):
    """Bind every public line SHA/count to its already-verified source sidecar.

    IDs/JSON grammar/privacy are freshly parsed/scanned by final cold; this
    narrow cache layer does not re-run the expensive sanitizer tree walk.
    """
    count=0
    with path.open('rb') as stream,side_path.open('rb') as identities:
        for count,line in enumerate(stream,1):
            identity=json.loads(identities.readline())
            require(identity['line']==count and identity['public_sha256']==hashlib.sha256(line).hexdigest()
                    and isinstance(identity.get('source_sha256'),str) and len(identity['source_sha256'])==64,
                    'R13 cache JSONL line/source-public sidecar differs')
        require(not identities.readline() and count==entry['line_count'],'R13 cache JSONL sidecar count differs')
    return count


def copy_verified_pair(old_public,relative,entry,target_public,side_entry=None):
    """Public helper for bounded negative controls; never mutates old files."""
    old_public,target_public=Path(old_public),Path(target_public)
    relative=safe_relative(relative);source=old_public/relative;target=target_public/relative
    require(source.is_file() and source.stat().st_size==entry['public_bytes']
            and sha(source)==entry['public_sha256'],'R13 cache old public body changed: '+relative)
    side=entry.get('line_identities_file');verified_lines=0
    if side:
        side=safe_relative(side)
        require(side_entry is not None,'R13 cache missing indexed JSONL identity body')
        path=old_public/side
        require(path.is_file() and path.stat().st_size==side_entry['public_bytes']
                and sha(path)==side_entry['public_sha256'],'R13 cache old sidecar changed')
        verified_lines=_verify_jsonl_lines(source,entry,path)
    pairs=[(relative,entry)] + ([(side,side_entry)] if side else [])
    for name,pin in pairs:
        destination=target_public/name
        require(not destination.exists(),'R13 cache target collision: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(old_public/name,destination)
        require(destination.stat().st_size==pin['public_bytes'] and sha(destination)==pin['public_sha256'],
                'R13 cache copied body readback differs: '+name)
    return verified_lines


class PinnedExportReuse:
    def __init__(self,stage,config):
        self.stage,self.config=Path(stage).resolve(),Path(config).resolve()
        self.config_sha256=sha(self.config)
        obj=json.loads(self.config.read_text(encoding='utf-8-sig'))
        require(obj.get('schema')==SCHEMA and obj.get('root_reviewed') is True
                and obj.get('old_failure_kind')=='semantic_original_vs_public_byte_size'
                and obj.get('fresh_full_cold_required') is True
                and obj.get('scientific_1396_results_unchanged') is True,
                'R13 exact Root-reviewed failed-export epoch required')
        require(Path(obj['old_stage']).resolve()==self.stage,'R13 cache stage/config path differs')
        require(set(obj['objects'])==set(OBJECT_NAMES),'R13 cache must bind scope/index/ZIP/audit/actual trace')
        paths={name:_read_pinned(item) for name,item in obj['objects'].items()}
        self.public=self.stage/'tree'/NAMESPACE
        require(paths['index'].samefile(self.public/'PUBLIC_CONTENT_REFS.json')
                and paths['audit'].samefile(self.stage/'PRIVATE_EXPORT_AUDIT.json')
                and paths['archive'].parent.samefile(self.stage),'R13 cache objects do not identify the same old stage')
        self.index=json.loads(paths['index'].read_text(encoding='utf-8-sig'))
        self.scope=json.loads(paths['sealed_scope'].read_text(encoding='utf-8-sig'))
        audit=json.loads(paths['audit'].read_text(encoding='utf-8-sig'))
        trace=paths['failure_trace'].read_text(encoding='utf-8-sig')
        require(self.index.get('schema')=='trajkv.round13.public_content_refs.v1'
                and self.index.get('namespace')==NAMESPACE and self.index.get('fixture_only') is False
                and self.index.get('source_scope_sha256')==obj['objects']['sealed_scope']['sha256'],
                'R13 old index/scope identity differs')
        require(self.scope.get('schema')=='trajkv.round13.public_scope.v1'
                and self.scope.get('fixture_only') is False and self.scope.get('stage_state')=='ROOT_SEALED'
                and self.scope['root_seal']['source_payload_sha256']==canonical({k:v for k,v in self.scope.items() if k!='root_seal'}),
                'R13 old source Root seal differs')
        require(audit.get('status')=='PASS_LOCAL_R13_EXPORT' and audit.get('unsafe')==[]
                and audit.get('selected_files')==len(self.scope['files'])
                and not (self.stage/'DELIVERY.json').exists(),'R13 cache requires preserved failed stage, not completed delivery')
        require('R13 readback byte count differs' in trace and 'semantic_callback' in trace
                and 'verify_public_round13_v1.py' in trace and 'round13_public_adapter_v1.py' in trace,
                'R13 actual trace must identify post-body semantic size failure')
        self.old_source={item['public_path']:item for item in self.scope['files']}
        require(len(self.old_source)==len(self.scope['files']),'R13 old scope paths duplicate')
        for relative,item in self.old_source.items():
            safe_relative(relative)
            require(relative in self.index['files']
                    and self.index['files'][relative]['source_sha256']==item['sha256'],
                    'R13 old original-source/index binding differs')
        self.reused=[];self.regenerated=[];self.reused_bytes=0;self.verified_lines=0

    def copy(self,relative,new_source_sha256,selected,target_public):
        # The scanner is always handled by the reviewed exact-scanner function.
        if relative=='ops/export_public.py' or relative not in self.old_source:
            self.regenerated.append(relative);return None
        old=self.old_source[relative];entry=self.index['files'][relative]
        if old['sha256']!=new_source_sha256 or entry['source_sha256']!=new_source_sha256:
            self.regenerated.append(relative);return None
        path=Path(selected['path']);stat=path.stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)==selected['stat']
                and sha(path)==new_source_sha256,'R13 cache actual new source no longer matches sealed SHA')
        side=entry.get('line_identities_file')
        side_entry=self.index['files'].get(side) if side else None
        self.verified_lines+=copy_verified_pair(self.public,relative,entry,target_public,side_entry)
        stat=path.stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)==selected['stat'],
                'R13 cache new source changed while copying')
        result=deepcopy(entry)
        result.update(source_origin=selected['origin'],source_path=selected['source_path'],source_bytes=stat.st_size)
        self.reused.append(relative);self.reused_bytes+=entry['public_bytes']+(side_entry['public_bytes'] if side_entry else 0)
        return result,side,[]

    def receipt(self):
        return {'schema':'trajkv.round13.export_cache_reuse_receipt.v1','root_pin_config_sha256':self.config_sha256,
            'old_stage':self.stage.name,'old_source_scope_sha256':self.index['source_scope_sha256'],
            'reused_source_files':len(self.reused),'regenerated_source_files':len(self.regenerated),
            'copied_verified_public_bytes_including_sidecars':self.reused_bytes,
            'JSONL_public_line_SHA_sidecar_bindings_rechecked':self.verified_lines,
            'reused_paths':self.reused,'regenerated_paths':self.regenerated,
            'original_old_files_mutated':False,'old_semantic_failure_not_promoted_to_PASS':True,
            'cache_is_not_fresh_full_cold_verification':True,'fresh_full_cold_required':True,
            'new_GPU_calls':0,'GitHub_mutation':False}
