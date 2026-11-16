"""CPU-only exact K2-reference layout and actual-owner controls."""
import argparse
import hashlib
import json
from pathlib import Path

from round6_depth_codec import plan_depth_stages
from round7_key_layout import plan_key_stages


def controls():
    layers, heads, dimension = 28, 4, 128
    # Interleave both axes so K coverage changes if a K1 arm recomputes oldcap.
    ranking = [(axis, layer, head) for layer in range(layers) for head in range(heads) for axis in ('k', 'v')]
    rows = []
    for nv in (128, 256, 768, 1792, 2048, 4096):
        full = layers * heads * nv * 512
        k2_base = layers * heads * nv * 50 + layers * heads * dimension * 6 + 2048 + nv * 8
        k1_base = layers * heads * nv * 34 + layers * heads * dimension * 6 + 128 + nv * 8
        for policy in ('incumbent', 'v_pair', 'v_mixed'):
            common = dict(layers=layers, heads=heads, visual_tokens=nv, visual_full_bytes=full,
                          factor_bytes=1344, budget_bits=1.8, selected_units=ranking, depth_policy=policy)
            old = plan_depth_stages(**common, base_bytes=k2_base)
            new = plan_key_stages(**common, base_bytes=k1_base, layout_reference='k2',
                                  key_index_bytes=16, book_bytes=128, reference_factor_bytes=1344)
            if new['selected_stages'] != old['selected_stages']:
                raise RuntimeError('A K1 image failed to preserve its exact K2 reference stages')
            if new['ranking_constant'] != old['ranking_constant'] or len(new['ranking_constant']) != 448:
                raise RuntimeError('The exact single complete named ranking was not retained')
            if len(new['policy_constant']) != 5 or new['policy_constant'][0] != old['policy_constant'][0] | 128 or new['policy_constant'][1:] != old['policy_constant'][1:]:
                raise RuntimeError('Reference flag was not stored inside the existing five-byte policy')
            saving = layers * heads * nv * 16 + 1920
            if old['predicted_bytes'] - new['predicted_bytes'] != saving:
                raise RuntimeError('Reference layout reused or misstated the K2 byte total')
            if new['predicted_bytes'] > new['allowed_bytes']:
                raise RuntimeError('Actual complete K1 owner exceeds the requested cap')
            actual = plan_key_stages(**common, base_bytes=k2_base)
            if actual != old:
                raise RuntimeError('Default no-reference K2 planning changed the legacy plan')
            rows.append(dict(nv=nv, policy=policy, selected_stages=len(new['selected_stages']),
                same_stage_tuples=True, old_complete_bytes=old['predicted_bytes'], actual_K1_complete_bytes=new['predicted_bytes'],
                actual_K1_bits=new['actual_budget_bits'], saving_bytes=saving,
                reference_affordable=new['layout_reference_affordable_stages'], actual_affordable=new['layout_actual_affordable_stages'],
                ranking_bytes=len(new['ranking_constant']), policy_bytes=len(new['policy_constant'])))
    common = dict(layers=layers, heads=heads, visual_tokens=1792,
        visual_full_bytes=layers * heads * 1792 * 512, factor_bytes=1344,
        base_bytes=layers * heads * 1792 * 50 + layers * heads * 128 * 6 + 2048 + 1792 * 8,
        budget_bits=1.8, selected_units=ranking)
    # Additional paid encoder constants must not be hidden by preserving a mask.
    rejected = False
    try:
        plan_key_stages(**dict(common, factor_bytes=1344 + 100000), layout_reference='k2', reference_factor_bytes=1344)
    except RuntimeError:
        rejected = True
    if not rejected:
        raise RuntimeError('An unaffordable preserved layout hid paid extra encoder constants')
    return dict(status='PASS', cases=rows, paid_constant_over_cap_rejected=True,
                scope='CPU exact allocation and complete logical-owner arithmetic only; no GPU/task result')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = controls()
    report['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('round7_key_layout.py', 'round7_key_bank.py', 'round7_key_layout_check.py')}
    (args.output / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], cases=len(report['cases']))), flush=True)


if __name__ == '__main__':
    main()
