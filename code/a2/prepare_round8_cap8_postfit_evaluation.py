"""Prepare a new cap8-aligned 11-arm evaluation, preserving old generators.

Reuses the existing endpoint/owner contracts in memory, but pins the new cap8
collector config, dump and eight books. No model, SSH or GPU is executed.
Plan-only is available before collection/fitting; actual QA generation requires
matching dump.config_sha256 plus endpoint provenance. Existing files cannot be
overwritten. Old cap16-to-cap8 results retain their cross-density scope.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import prepare_round8_postfit_evaluation as base


ROOT = base.ROOT
COLLECTOR = ROOT/'configs/a2/round8_v1/native_collect_cal32_select32_cap8_v2.json'
FIT_PLAN = ROOT/'configs/a2/round8_v1/native_fit_plan_cap8_v2.json'
RUNTIME_JOB = ROOT/'configs/jobs/r8_native_collect_cal32_select32_cap8_gpu0_v2.json'
DEFAULT_PREFIX = 'postfit_cap8_aligned11_v3'
STATE = {}


def capture_contract(a,c,require_complete=False):
    collector = base.read(COLLECTOR)
    base.require(collector['arguments']['max_patches'] == 8,'Cap8 collector declaration required')
    base.require(collector['selected_stages'] == c['baseline']['selected_stages'][:43], 'Cap8 mask differs')
    path = a.dump_manifest or base.local_path(c['fit_plan']['dump'])/'manifest.json'
    result = dict(calibration_max_patches=8,quality_arguments_max_patches=8,quality_variant_cap=8,
        collector_config=str(COLLECTOR.relative_to(ROOT)),collector_config_sha256=base.sha(COLLECTOR),
        native_dump_manifest=str(path),native_dump_manifest_sha256=None,
        source_density_alignment_verified=False,
        previous_run_scope='old cap16 calibration to cap8 quality is valid cross-density development, not matched-cap calibration evidence')
    if not path.is_file():
        base.require(not require_complete,'Cap8 native manifest is pending')
        result['status']='CAP8_DECLARED_COLLECTION_AND_FITS_PENDING'
        return result
    dump = base.read(path)
    base.require(dump['config_sha256'] == base.sha(COLLECTOR), 'Native dump did not consume the exact cap8 collector config')
    expected_sources = set(c['split']['train'])|set(c['split']['select'])
    entries = dump['entries']
    base.require(len(entries) == 64 and {row['source_id'] for row in entries} == expected_sources,
        'Cap8 native source union differs from fixed train32/select32')
    expected_inputs = {Path(name).name:base.sha(base.local_path(name)) for name in collector['sample_paths']}
    base.require({row['name']:row['sha256'] for row in dump['sample_inputs']} == expected_inputs,
        'Cap8 native stimulus bytes differ from fixed source manifests')
    result.update(native_dump_manifest_sha256=base.sha(path),
        expected_native_visual_tokens_by_source={row['source_id']:row['visual_tokens'] for row in entries},
        status='CAP8_COLLECTION_CONFIG_AND_SOURCE_MEMBERSHIP_VERIFIED')
    return result


def install_adapters():
    """Narrow adapters do not edit the old generator or any prior output."""
    base.FIT_PLAN,base.RUNTIME_JOB = FIT_PLAN,RUNTIME_JOB
    original_context,original_verify,original_variants,original_new = base.load_context,base.verified_books,base.make_variants,base.new

    def context(a):
        base.require(a.output_prefix.startswith('postfit_cap8_aligned'),'Use a new cap8-aligned output prefix')
        c = original_context(a)
        c['smoke'] = copy.deepcopy(c['smoke']); c['smoke']['arguments']['max_patches'] = 8
        c['baseline_config'] = copy.deepcopy(c['baseline_config']); c['baseline_config']['arguments']['max_patches'] = 8
        c['baseline'] = copy.deepcopy(c['baseline']); c['baseline']['cap'] = 8
        c['template'] = copy.deepcopy(c['template']); c['template']['cap'] = 8
        for variant in c['baseline_config']['variants']:
            variant['cap'] = 8
        STATE.update(context=c,alignment=capture_contract(a,c))
        return c

    def expected_paths(a,c):
        result = {}
        for axis in ('k','v'):
            remote = c['fit_plan']['jobs'][axis.upper()]['output']
            local = (a.k_fit_dir if axis == 'k' else a.v_fit_dir) or base.local_path(remote)
            assets = c['fit_plan']['jobs'][axis.upper()]['assets']
            result[axis] = dict(local=str(local),remote=remote,complete=str(local/'complete_books.json'),
                files=[asset+'_'+mode.replace('-','_')+'.json' for asset in assets for mode in c['fit_plan']['modes']])
        return result

    def verified(a,c,expected):
        alignment = capture_contract(a,c,require_complete=True)
        assets,receipts = original_verify(a,c,expected)
        base.require(all(row['provenance']['dump_manifest_sha256'] == alignment['native_dump_manifest_sha256']
            for row in receipts.values()),'Fitted endpoints did not consume the verified cap8 dump')
        alignment['source_density_alignment_verified'] = True
        alignment['status'] = 'CAP8_COLLECTION_AND_EIGHT_ENDPOINTS_VERIFIED'
        STATE['alignment'] = alignment
        return assets,receipts

    def variants(a,c,assets):
        rows,notes = original_variants(a,c,assets)
        for row in rows:
            row['cap'] = 8
        base.require(len(rows) == 11,'Cap8 repair preserves the same eleven-arm comparison')
        notes['cap_alignment']='Collector maximum patches8; QA arguments.max_patches8; every variant.cap8; exact new dump and fitted-book provenance checked'
        return rows,notes

    def new(path,value):
        value = copy.deepcopy(value)
        if isinstance(value,dict):
            value['cap_alignment_contract'] = copy.deepcopy(STATE['alignment'])
            value['cap8_preparer_source_sha256'] = base.sha(__file__)
            if 'variants' in value and 'arguments' in value:
                value['arguments']['max_patches'] = 8
                for row in value['variants']:
                    row['cap'] = 8
                base.require(value['cap_alignment_contract']['source_density_alignment_verified'] is True,
                    'Do not create QA configs before cap8 endpoints are verified')
            if 'argv' in value and 'round8_source_root' in value:
                cache = STATE['context']['global_root']+'/cache/round8_v1/shared'
                for key,leaf in [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),
                    ('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('PYTHONPYCACHEPREFIX','pycache'),
                    ('TORCH_HOME','torch'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('PIP_CACHE_DIR','pip'),
                    ('UV_CACHE_DIR','uv'),('XDG_CACHE_HOME','xdg')]:
                    value['env'][key] = cache+'/'+leaf
                value['prerequisites'].append('arguments.max_patches8 and every variant.cap8; runtime Nv must match the cap8 native source manifest')
        original_new(path,value)

    base.load_context,base.expected_paths,base.verified_books,base.make_variants,base.new = context,expected_paths,verified,variants,new


def main():
    if not any(arg in ('--name','--output-prefix') or arg.startswith('--name=') or arg.startswith('--output-prefix=') for arg in sys.argv[1:]):
        sys.argv.extend(['--name',DEFAULT_PREFIX])
    install_adapters()
    base.main()


if __name__ == '__main__':
    main()
