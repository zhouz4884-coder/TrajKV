"""CPU-only structural checks for the new first-question adapter."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from round6_second_session import (bank_layer_proxy, decoder_layers,
    first_question_features, image_token_id, model_geometry)

ROOT = Path(__file__).resolve().parents[2]


def main():
    checks = []
    layers = [NS(self_attn=NS(q_proj=NS(out_features=4096),
                            k_proj=NS(out_features=4096), v_proj=NS(out_features=4096))) for _ in range(32)]
    config = NS(model_type='llava', image_token_index=32000,
                text_config=NS(num_hidden_layers=32, num_attention_heads=32,
                               num_key_value_heads=32, hidden_size=4096))
    modern = NS(model=NS(language_model=NS(layers=layers)), config=config)
    old = NS(language_model=NS(model=NS(layers=layers)), config=config)
    for model, expected in [(modern, 'model.language_model.layers'), (old, 'language_model.model.layers')]:
        actual, path = decoder_layers(model)
        assert path == expected and actual == layers
        assert bank_layer_proxy(model).model.language_model.layers == layers
        geometry = model_geometry(model)
        assert (geometry['layers'], geometry['query_heads'], geometry['kv_heads'], geometry['head_dim']) == (32, 32, 32, 128)
        checks.append('layer/geometry/proxy: ' + path)
    assert image_token_id(modern) == 32000
    assert not hasattr(config, 'image_token_id')  # alias lookup never mutates model
    checks.append('image_token_index alias without model mutation')
    conflict = NS(config=NS(image_token_id=12, image_token_index=13))
    try:
        image_token_id(conflict)
    except ValueError:
        checks.append('conflicting image token aliases rejected')
    else:
        raise AssertionError('Conflicting aliases accepted')
    row = json.loads((ROOT / 'configs/a2/docvqa_next_dev64.json').read_text(encoding='utf-8'))[0]
    row = dict(row, remote_images=row['images'])

    class Processor:
        chat_template = 'official local template'
        tokenizer = NS(chat_template=None)

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            assert messages[0]['content'][1]['text'] == row['questions'][0]
            return 'USER: <image>\n' + row['questions'][0] + ' ASSISTANT:'

        def __call__(self, **kwargs):
            assert set(kwargs) == {'text', 'images', 'return_tensors'}
            assert len(kwargs['images']) == 1
            return {'input_ids':torch.tensor([[1] + [32000] * 576 + [2, 3]]),
                    'pixel_values':torch.zeros(1, 3, 336, 336)}

    features, visual, metadata = first_question_features(Processor(), modern, row)
    assert visual.numel() == 576 and metadata['padded_visual_tokens'] == 0
    assert metadata['turns'] == 1 and features['input_ids'].shape[-1] == 579
    checks.append('official template; exactly 576 real tokens; no padding/InternVL kwargs')
    fallback = Processor()
    fallback.chat_template = None
    _, visual, metadata = first_question_features(fallback, modern, row)
    assert 'fallback' in metadata['prompt_protocol'] and visual.numel() == 576
    checks.append('LLaVA1.5 first-turn serialization fallback')
    unsupported = NS(model=modern.model, config=NS(model_type='llava_next', image_token_index=32000))
    try:
        first_question_features(fallback, unsupported, row)
    except ValueError:
        checks.append('LLaVA-next without official template rejected')
    else:
        raise AssertionError('Unsupported template accepted')
    output = dict(status='PASS_CPU_STRUCTURAL_ONLY', GPU_used=False, model_loaded=False,
        checks=checks, adapter_sha256=hashlib.sha256((ROOT / 'code/a2/round6_second_session.py').read_bytes()).hexdigest(),
        scope='mock decoder/processor plus actual old development image; no evidence of GPU identity or model task quality')
    target = ROOT / 'audit/a2/round6_second_adapter_check_v1.json'
    with target.open('x') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
