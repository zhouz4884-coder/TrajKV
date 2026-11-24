"""Exact arithmetic for the deployed R budget; no quality or GPU claims."""
import json
from pathlib import Path


def main():
    target = Path('${EVIDENCE_ROOT}/results/round12/ROOT_RESOURCE_ANALYSIS_v1.json')
    cases = []
    for n in (256, 768, 1280, 1792, 2304, 4096):
        g8_bytes = 5486 * n + 155145
        r_base = 3816 * n + 89737
        m = min(224, (g8_bytes - r_base) // (18 * n))
        r_bytes = r_base + m * 18 * n
        assert m >= 43 and r_bytes <= g8_bytes
        assert m == 224 or r_bytes + 18 * n > g8_bytes
        cases.append({'visual_tokens': n, 'R_residual_slots': m, 'R_visual_owner_bytes': r_bytes,
                      'G8_visual_owner_bytes': g8_bytes, 'unspent_bytes': g8_bytes - r_bytes,
                      'R_actual_bits_per_element': r_bytes / (3584 * n),
                      'G8_actual_bits_per_element': g8_bytes / (3584 * n)})
    result = {
        'schema': 'trajkv.round12.root_resource_analysis.v1',
        'source': 'code/round12_static_bank_v1.py expected_owner_bytes and mode R',
        'analysis_type': 'exact_integer_cost_arithmetic_only',
        'new_GPU_calls': 0,
        'new_task_answers': 0,
        'complete_R_owner_formula': '(3816 + 18*m)*N + 89737',
        'complete_G8_owner_formula': '5486*N + 155145',
        'R_selection_formula': 'min(224, floor((1670*N + 65408)/(18*N)))',
        'scope': 'maximum feasible prefix of an already frozen ordered list; not a task-quality optimum',
        'quality_claims': [],
        'reporting': 'R is approximately 1.55 mean bit, not a uniform per-image 1.55-bit cap; report all-row maximum separately',
        'cases': cases,
    }
    body = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == body
    else:
        target.write_text(body, encoding='utf-8')
    print(body)


if __name__ == '__main__':
    main()
