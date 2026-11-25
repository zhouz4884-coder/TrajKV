"""R13 v1 export body with a narrowly Root-pinned unchanged-body writer.

The frozen v1 file and its module globals are not modified. Its exact export
function code is invoked with a private globals dictionary that replaces only
the original-body writer/exact-scanner accounting. Generated scope/phase/index,
ZIP construction and the full fresh cold verification remain the v1 logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

import export_public_round13_v1 as frozen
from round13_export_cache_v1 import PinnedExportReuse

FROZEN_V1_SHA='fa3bb3cfda26dc61f36cc6ea64db76d6d239bc9e3f0fbc8472e01f6c925678bd'


def export(scope_path,stage,tag,*,semantic_verifier,reuse_stage,reuse_config,semantic_callback=None):
    frozen.require(frozen.sha(Path(frozen.__file__))==FROZEN_V1_SHA,'R13 frozen v1 exporter source changed')
    scope_path=Path(scope_path);scope=json.loads(scope_path.read_text(encoding='utf-8-sig'))
    source={item['public_path']:item for item in scope['files']}
    cache=PinnedExportReuse(reuse_stage,reuse_config)
    frozen.require(any(Path(item['source_path']).resolve()==cache.config and item['sha256']==cache.config_sha256
                       for item in scope['files']),'Root reuse pin config must be part of new sealed source scope')
    for name in ('export_public_round13_v2.py','round13_export_cache_v1.py'):
        relative='ops/'+name
        frozen.require(relative in source and source[relative]['sha256']==frozen.sha(Path(__file__).with_name(name)),
                       'R13 new cache/export source must be in sealed scope: '+relative)
    original_writer=frozen.write_public_file;original_exact=frozen.exact_scanner
    def writer(relative,selected,public,sanitizer):
        reused=cache.copy(relative,source[relative]['sha256'],selected,public)
        return reused if reused is not None else original_writer(relative,selected,public,sanitizer)
    def exact(relative,selected,public):
        cache.regenerated.append(relative)
        return original_exact(relative,selected,public)
    private_globals=dict(frozen.export.__globals__)
    private_globals.update(write_public_file=writer,exact_scanner=exact)
    operation=types.FunctionType(frozen.export.__code__,private_globals,'root_pinned_r13_export_v2',
                                 frozen.export.__defaults__,frozen.export.__closure__)
    operation.__kwdefaults__=dict(frozen.export.__kwdefaults__ or {})
    receipt=None
    try:
        delivery=operation(scope_path,stage,tag,semantic_verifier=semantic_verifier,semantic_callback=semantic_callback)
        receipt={**cache.receipt(),'status':'PASS_REUSE_AND_FRESH_FULL_COLD_NOT_PUBLISHED',
            'fresh_full_cold_status':delivery['cold_verification_status'],
            'frozen_exporter_v1_sha256':FROZEN_V1_SHA,
            'exporter_v2_sha256':frozen.sha(__file__),
            'cache_helper_sha256':frozen.sha(Path(__file__).with_name('round13_export_cache_v1.py')),
            'frozen_module_globals_modified':False,'scientific_results_modified':False}
        return delivery
    except Exception as exc:
        receipt={**cache.receipt(),'status':'EXPORT_FAILED_PRESERVED_NOT_PUBLICATION_PASS',
            'error_type':type(exc).__name__,'sensitive_error_text_omitted':True,
            'frozen_module_globals_modified':False,'scientific_results_modified':False}
        raise
    finally:
        target=frozen.EVIDENCE/'public_staging'/stage
        if target.is_dir() and receipt is not None:
            frozen._write_new(target/'EXPORT_REUSE_READBACK_RECEIPT_v1.json',receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--tag',required=True);p.add_argument('--semantic-file',required=True)
    p.add_argument('--semantic-function',default='verify_semantics');p.add_argument('--reuse-stage',type=Path,required=True)
    p.add_argument('--reuse-config',type=Path,required=True);a=p.parse_args()
    result=export(a.scope,a.stage,a.tag,semantic_verifier={'relative_path':a.semantic_file,'function':a.semantic_function},
                  reuse_stage=a.reuse_stage,reuse_config=a.reuse_config)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
