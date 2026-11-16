"""Round-6 逐题翻转算术：把 pp 还原成"翻了几道题"。

不读任何张量、不跑 GPU，只读两份 scores.csv 与一份单轴编码阶梯汇总。
输出（默认写到 results/round6_v1/flip_arithmetic/）：
  summary.json        每对比较的 Δ / SD / SE / 半宽 / 翻转计数 / 设计效应
  strata.csv          按 incumbent 分数分层的"被任一臂翻过"比例
  ladder.csv          单轴编码阶梯（K 与 V 各自换轴后的 ANLS）
  union.csv           每个题被几个臂翻过

用法：
  python code/a2/analyze_round6_flip_arithmetic.py
"""
import collections
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'results/round6_v1/doc_confirmation_analysis/scores.csv'
TEXT = ROOT / 'results/round6_v1/text_confirmation_analysis/scores.csv'
LADDER = ROOT / 'results/a2_analysis_value/summary.json'
OUT = ROOT / 'results/round6_v1/flip_arithmetic'

BASE = 'r6_depth_incumbent'


def load(path, cluster_key):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    for row in rows:
        row['score'] = float(row['score'])
    by = collections.defaultdict(dict)
    for row in rows:
        by[(row[cluster_key], row['session_id'], row['turn'])][row['variant']] = row
    return by


def pair_stats(by, a, b, cluster_index=0):
    keys = [k for k in by if a in by[k] and b in by[k]]
    delta = [by[k][a]['score'] - by[k][b]['score'] for k in keys]
    n = len(keys)
    mean = statistics.mean(delta)
    var = statistics.variance(delta) if n > 1 else 0.0
    se = math.sqrt(var / n)
    hw = 1.96 * se
    up = sum(1 for x in delta if x > 1e-9)
    down = sum(1 for x in delta if x < -1e-9)
    # 文档内相关与设计效应（Kish 校正）
    groups = collections.defaultdict(list)
    for k in keys:
        groups[k[cluster_index]].append(by[k][a]['score'] - by[k][b]['score'])
    g = len(groups)
    m_bar = n / g
    grand = mean
    ssb = sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in groups.values())
    ssw = sum((x - statistics.mean(v)) ** 2 for v in groups.values() for x in v)
    msb = ssb / max(g - 1, 1)
    msw = ssw / max(n - g, 1)
    icc = max(0.0, (msb - msw) / (msb + (m_bar - 1) * msw)) if msb or msw else 0.0
    deff = 1 + (m_bar - 1) * icc
    net_units = mean * n
    return dict(
        a=a, b=b, n=n, delta_pp=100 * mean, sd=math.sqrt(var),
        se_pp=100 * se, half_width_pp=100 * hw,
        flipped=int(up + down), flipped_up=int(up), flipped_down=int(down),
        net_score_units=net_units,
        half_width_over_point=(hw / abs(mean)) if abs(mean) > 1e-12 else None,
        sign_test_p=_sign_test(up, down),
        documents=g, questions_per_cluster=m_bar, icc=icc, design_effect=deff,
        half_width_pp_clustered=100 * 1.96 * math.sqrt(var * deff / n),
    )


def _sign_test(up, down):
    total = up + down
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(0, min(up, down) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def stratum_table(by, variants):
    rows = []
    for key, cell in by.items():
        if BASE not in cell:
            continue
        hits = sum(1 for v in variants
                   if v in cell and abs(cell[v]['score'] - cell[BASE]['score']) > 1e-9)
        score = cell[BASE]['score']
        label = 'perfect' if score > 0.999 else ('zero' if score == 0 else 'partial')
        rows.append((key, label, hits))
    out = []
    for label in ('perfect', 'partial', 'zero'):
        n = sum(1 for _, l, _ in rows if l == label)
        f = sum(1 for _, l, h in rows if l == label and h > 0)
        out.append(dict(stratum=label, questions=n, flipped_by_any_arm=f,
                        rate=100 * f / n if n else 0.0))
    hist = collections.Counter(h for _, _, h in rows)
    return out, hist, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, path, cluster in (('doc54', DOC, 'source_document_id'),
                                ('text256', TEXT, 'source_image_id')):
        by = load(path, cluster)
        variants = sorted({v for cell in by.values() for v in cell})
        others = [v for v in variants if v != BASE]
        stats = [pair_stats(by, v, BASE) for v in others]
        # 同字节臂之间的关键一对
        if 'r6_axis_matched_spread' in variants:
            stats.append(pair_stats(by, 'r6_axis_matched_spread', 'r6_depth_v_pair'))
        n = sum(1 for k in by if BASE in by[k])
        strata, hist, rows = stratum_table(by, others)
        union = sum(1 for _, _, h in rows if h > 0)
        jaccard = []
        flipped = {v: {k for k, cell in by.items()
                       if BASE in cell and v in cell
                       and abs(cell[v]['score'] - cell[BASE]['score']) > 1e-9}
                   for v in others}
        for i, a in enumerate(others):
            for b in others[i + 1:]:
                inter = flipped[a] & flipped[b]
                uni = flipped[a] | flipped[b]
                jaccard.append(dict(a=a, b=b, shared=len(inter),
                                    jaccard=len(inter) / len(uni) if uni else 0.0))
        jaccard.sort(key=lambda x: -x['jaccard'])
        report[name] = dict(n=n, question_quantum_pp=100 / n, pairs=stats,
                            strata=strata, union_flipped=union,
                            union_rate=100 * union / n,
                            flip_multiple_histogram=dict(sorted(hist.items())),
                            jaccard_top=jaccard[:5])
    # 单轴编码阶梯：K 与 V 各自单独降精度（另一轴 FP16）
    ladder = []
    if LADDER.exists():
        data = json.loads(LADDER.read_text())
        for row in data['table']:
            ladder.append(dict(variant=row['variant'], questions=row['questions'],
                               anls=row['anls'], exact=row['exact']))
    report['ladder'] = dict(source=str(LADDER.relative_to(ROOT)),
                            selection='first 20 reference answers with ASCII digits, 9 images',
                            rows=ladder)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'pairs' and kk != 'rows'}
                      for k, v in report.items()}, indent=1, ensure_ascii=False)[:1200])
    for name in ('doc54', 'text256'):
        print(f"\n== {name}  n={report[name]['n']}  1 题 = {report[name]['question_quantum_pp']:.3f} pp")
        print(f"{'comparison':44s}{'d_pp':>8s}{'hw':>7s}{'hw/|d|':>7s}{'k':>4s}{'(+/-)':>8s}{'net':>7s}{'DEFF':>6s}{'p_sign':>8s}")
        for s in report[name]['pairs']:
            print(f"{s['a'][:20]+' vs '+s['b'][:18]:44s}{s['delta_pp']:+8.3f}{s['half_width_pp']:7.2f}"
                  f"{(s['half_width_over_point'] or 0):7.2f}{s['flipped']:4d}"
                  f"{str(s['flipped_up'])+'/'+str(s['flipped_down']):>8s}{s['net_score_units']:+7.2f}"
                  f"{s['design_effect']:6.2f}{s['sign_test_p']:8.3f}")
        print('  strata:', [(s['stratum'], s['questions'], f"{s['rate']:.1f}%") for s in report[name]['strata']])
        print(f"  union flipped = {report[name]['union_flipped']}/{report[name]['n']} ({report[name]['union_rate']:.1f}%)")
        print('  top jaccard:', [(j['a'][:18], j['b'][:18], round(j['jaccard'], 2)) for j in report[name]['jaccard_top'][:3]])
    if ladder:
        print('\n== 单轴编码阶梯（other axis = FP16，n=20/9 图）')
        for row in ladder:
            print(f"  {row['variant']:38s} anls={100*row['anls']:6.2f}  exact={100*row['exact']:6.2f}")


if __name__ == '__main__':
    main()
