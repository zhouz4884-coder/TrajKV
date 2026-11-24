"""Fresh CPU AutoProcessor input bindings; zero model loads/forwards/logits.

Run only after a new confirmation dataset and selection decision are frozen.
No prior native prediction/hash artifact is accepted as an input.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import round12_confirmation_admission_v1 as admission
from model_registry import get_model_spec

SCHEMA = admission.CPU_BINDING_CONFIG_SCHEMA
RUNTIME_FILES = admission.CPU_BINDING_SOURCE_FILES
M1_PROTOCOL_SHA = '4ba12190fe3c6513c4a45822c4294c3e40cf82ea13ac791c6cb677aed3871427'
sha, read, require = admission.sha, admission.read, admission.require


def runtime_source_manifest():
    require(sha(ROOT / 'code/m1_protocol.py') == M1_PROTOCOL_SHA, 'frozen native M1 processor protocol changed')
    return {'code/' + name: sha(ROOT / 'code' / name) for name in RUNTIME_FILES}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def processor_file_identity(config):
    """Verify official config/tokenizer files; deliberately do not read weights."""
    model = Path(config['model_path']).resolve()
    lock_path = Path(config['model_source_lock'])
    spec = get_model_spec('M1')
    raw = read(lock_path)
    lock = raw if 'siblings' in raw else raw[spec['hf_id']]
    require(lock.get('revision', lock.get('sha')) == admission.MODEL_REVISION, 'processor checkpoint tree revision differs')
    files = []
    for entry in lock['siblings']:
        name = entry['rfilename']
        if '/' in name or not name.endswith(('.json', '.txt', '.jinja', '.model')):
            continue
        p = model / name
        require(p.is_file() and p.stat().st_size == entry['size'], 'processor file size/missing: ' + name)
        body = p.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if entry.get('lfs'):
            require(digest == entry['lfs']['sha256'], 'processor LFS identity differs: ' + name)
        else:
            blob = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            require(blob == entry['blobId'], 'processor Git blob differs: ' + name)
        files.append({'file': name, 'sha256': digest, 'bytes': len(body)})
    require(files, 'no official processor files bound')
    return {'model_revision': spec['revision'], 'model_source_lock_sha256': sha(lock_path),
            'files': files, 'weight_files_read_or_loaded': False}


def build_bindings(config, output, *, config_sha256=None):
    require(config.get('schema') == SCHEMA, 'CPU binding config schema differs')
    require(config.get('native_bindings_path') is None and config.get('native_bindings_sha256') is None,
            'CPU maker cannot consume old native bindings')
    identity = admission.validate_data_config(config, require_bindings=False)
    project = Path(config['project_root']).resolve()
    require(os.name != 'nt' and str(project).startswith('/media/') and project.name == 'baseline_matrix_v3',
            'actual CPU bindings must be generated on the named server data-disk project')
    output = Path(output).resolve()
    require(output.is_relative_to(project) and not output.exists(), 'new binding output must be on project data disk')
    require(Path(config['model_path']).resolve().is_relative_to(project.parent), 'processor model path escapes data workspace')
    cache = Path(config['cache_root']).resolve()
    require(cache.is_relative_to(project), 'CPU cache must be on project data disk')
    for key in ('TMPDIR', 'TMP', 'TEMP', 'HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME', 'PYTHONPYCACHEPREFIX'):
        p = cache / key.lower()
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    sources = runtime_source_manifest()
    require(config.get('expected_source_hashes') == sources, 'CPU binding source graph differs')
    official = processor_file_identity(config)
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from transformers import AutoProcessor
    from m1_protocol import prepare_m1_inputs
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
    classes = [type(processor), type(processor.tokenizer), type(processor.image_processor)]
    library_sources = {str(Path(inspect.getfile(c))): sha(inspect.getfile(c)) for c in classes}
    spec = get_model_spec('M1')
    bindings = {}
    for index, row in enumerate(identity['rows']):
        require(sha(row['image_path']) == row['original_image_sha256'], 'new source image bytes changed')
        with Image.open(row['image_path']) as original:
            image = original.convert('RGB')
        rendered = row['question'] + admission.SUFFIX
        batch = prepare_m1_inputs(processor, image, rendered, image_token_id=spec['image_token_id'],
                                  max_patches=8, device='cpu')
        alternative_question = 'What is visible in the image?' if row['question'] == 'Describe the objects.' else 'Describe the objects.'
        alternate = prepare_m1_inputs(processor, image, alternative_question + admission.SUFFIX,
                                      image_token_id=spec['image_token_id'], max_patches=8, device='cpu')
        require(batch.boundary == alternate.boundary
                and torch.equal(batch.ids[:, :batch.boundary], alternate.ids[:, :alternate.boundary])
                and torch.equal(batch.visual_positions, alternate.visual_positions),
                'new CPU visual prefix depends on future question')
        for key in set(batch.inputs) | set(alternate.inputs):
            if key in ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids'):
                continue
            a, b = batch.inputs[key], alternate.inputs[key]
            require(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b,
                    'processor image values depend on future question')
        template = processor.chat_template
        template_sha = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else admission.data.canonical_sha(template)
        pixels = batch.inputs.get('pixel_values')
        require(isinstance(pixels, torch.Tensor) and pixels.device.type == 'cpu'
                and tuple(pixels.shape[1:]) == (3, 448, 448)
                and int(batch.visual_positions.numel()) == 256 * int(pixels.shape[0]), 'actual native tile geometry differs')
        bindings[row['id']] = {
            'id': row['id'], 'original_image_sha256': row['original_image_sha256'],
            'raw_question_sha256': row['question_sha256'], 'rendered_question_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'full_prompt_hash': hashlib.sha256(batch.prompt.encode()).hexdigest(), 'chat_template_hash': template_sha,
            'input_ids_sha256': tensor_sha(batch.ids), 'input_token_ids': batch.ids.reshape(-1).tolist(),
            'question_suffix_sha256': tensor_sha(batch.ids[:, batch.boundary:]),
            'native_position_ids_sha256': tensor_sha(batch.positions),
            'native_position_ids_shape': list(batch.positions.shape), 'native_position_ids': batch.positions.reshape(-1).tolist(),
            'compression_boundary': batch.boundary, 'actual_visual_token_count': int(batch.visual_positions.numel()),
            'visual_positions': batch.visual_positions.tolist(), 'input_tokens': batch.full_length,
            'pixel_values_shape': list(pixels.shape), 'pixel_values_sha256': tensor_sha(pixels),
            'model_commit': admission.MODEL_REVISION, 'dtype': 'float16',
            'preprocessing': {'max_patches': 8}, 'question_unknown_prefix_verified': True,
            'position_scope': 'fresh processor-derived planned serial positions, not model logits evidence',
        }
        batch = alternate = image = pixels = a = b = None
        gc.collect()
        print(json.dumps({'event': 'CPU_NATIVE_BINDING', 'id': row['id'], 'completed': index + 1,
                          'expected': len(identity['rows']), 'model_calls': 0}), flush=True)
    require(runtime_source_manifest() == sources and processor_file_identity(config) == official, 'CPU binding sources/files changed')
    require(all(sha(path) == digest for path, digest in library_sources.items()), 'processor library source changed')
    require(sha(config['input_manifest_path']) == config['input_manifest_sha256']
            and sha(config['inputs']) == config['inputs_sha256']
            and sha(config['frozen_decision_path']) == config['frozen_decision_sha256'], 'data/decision changed during CPU preparation')
    result = {'schema': admission.CPU_BINDING_SCHEMA, 'status': 'COMPLETE_CPU_PROCESSOR_BINDINGS_NO_MODEL',
        'model_id': 'M1', 'model_revision': admission.MODEL_REVISION,
        'input_manifest_sha256': config['input_manifest_sha256'], 'frozen_decision_sha256': config['frozen_decision_sha256'],
        'input_files': {Path(config['inputs']).name: {'sha256': config['inputs_sha256'], 'input_count': config['input_count']}},
        'bindings': bindings, 'bound_ids': list(bindings), 'question_suffix': admission.SUFFIX, 'preprocessing': {'max_patches': 8},
        'source_hashes': sources, 'processor_file_identity': official, 'processor_library_source_sha256': library_sources,
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'tokenizers_version': tokenizers.__version__,
        'config_sha256': config_sha256, 'model_loaded': False, 'model_forward_calls': 0,
        'QA': 0, 'old_logits_or_prediction_bindings_read': False,
        'bitexact_model_output_equivalence_claimed': False, 'CPU_preprocess_calls': 2 * len(bindings)}
    admission.data.freeze_json(output, result)
    return {'output': str(output), 'sha256': sha(output), 'bound_sources': len(bindings), 'QA': 0, 'model_loaded': False}


def config_template():
    value = admission.data_config_template()
    value.update(schema=SCHEMA, project_root=None, cache_root=None, expected_source_hashes=None,
                 status='UNCONFIGURED_NO_CONFIRMATION_SOURCES_SELECTED')
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--print-template', action='store_true')
    a = p.parse_args()
    if a.print_template:
        print(json.dumps(config_template(), ensure_ascii=False, indent=2))
        return
    require(a.config is not None and a.output is not None, 'config and new output required')
    body = a.config.read_bytes()
    result = build_bindings(json.loads(body.decode('utf-8-sig')), a.output,
                            config_sha256=hashlib.sha256(body).hexdigest())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
