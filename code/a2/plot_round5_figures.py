"""Publication-style round5 figures from complete, real summary tables.

The frozen audit selects displayed methods; confirmation/transfer results never
select the winner. Preview mode draws development only, with no empty or
fabricated confirmation panel. All plot data and input hashes are exported.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
PALETTE = dict(blue_main='#0F4D92', blue_secondary='#3775BA', teal='#42949E',
               red_strong='#B64342', neutral='#CFCECE', grey='#767676',
               dark='#4D4D4D', violet='#9A4D8E', green_1='#DDF3DE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if value.get('status') not in ('COMPLETE', 'PASS'):
        raise ValueError('A complete real summary is required: '+str(path))
    if not isinstance(value.get('table'), list) or not value['table']:
        raise ValueError('No measured table in '+str(path))
    return value


def publication_style():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in ('Arial', 'Helvetica', 'DejaVu Sans') if name in installed), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': chosen, 'font.size': 15, 'axes.labelsize': 14,
        'axes.titlesize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': 1.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 11,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'text.usetex': False, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    return chosen


def merge_development(paths):
    source, rows = None, {}
    for path in paths:
        item = load(path)
        if item.get('analysis_role') != 'development':
            raise ValueError('Development input must declare analysis_role=development')
        if source is None:
            source = item
        for field in ('images', 'questions_per_variant', 'samples_sha256'):
            if item.get(field) != source.get(field):
                raise ValueError('Cannot merge different development cohorts: '+field)
        for row in item['table']:
            name = row['variant']
            if name in rows:
                for field in ('anls', 'ci_low', 'ci_high', 'budget_bits_including_encoder_constants'):
                    if not math.isclose(float(rows[name][field]), float(row[field]), rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('Conflicting development rows for '+name)
            rows[name] = row
    merged = dict(source)
    merged['table'] = list(rows.values())
    return merged


def frozen_methods(audit):
    primary = audit['primary']
    secondary = audit['secondary_low_storage']
    finite = audit['original_finite_v']
    names = list(dict.fromkeys(['cold_meta_all_kv', secondary, finite, primary, 'base2_8', 'native4_8']))
    allowed = {row['name'] for row in audit['variants']}
    if not set(names+['full8']) <= allowed:
        raise ValueError('A displayed method was not in the frozen audit')
    caps = {row.get('cap', 8) for row in audit['variants'] if row['name'] in names+['full8']}
    if len(caps) != 1:
        raise ValueError('Displayed methods do not share the frozen vision cap')
    return names, primary, secondary, finite, next(iter(caps))


def descriptions(names, primary, secondary, finite):
    known = {
        'cold_meta_all_kv': ('K2-PQ/V1 base', PALETTE['grey'], 'o'),
        'base2_8': ('Base2', PALETTE['red_strong'], 'X'),
        'native4_8': ('Native4', PALETTE['dark'], 'P'),
    }
    known[secondary] = ('V residual, 20 units', PALETTE['teal'], '^')
    known[finite] = ('V residual, max budget', PALETTE['blue_secondary'], 's')
    known[primary] = ('Mixed-grad (frozen primary)', PALETTE['blue_main'], '*')
    return {name: known[name] for name in names}


def measured(row, metric):
    fields = [metric, 'ci_low', 'ci_high', 'budget_bits_including_encoder_constants']
    if any(field not in row for field in fields):
        raise ValueError('Missing measured metric/CI/actual-bit field: '+str(row.get('variant')))
    score, low, high, bits = [float(row[field]) for field in fields]
    if not all(math.isfinite(v) for v in (score, low, high, bits)):
        raise ValueError('Nonfinite plotted value')
    if not 0 <= low <= score <= high <= 1 or bits <= 0:
        raise ValueError('Expected fractional scores with an actual valid 95% interval')
    if row.get('materialized_quality_port', False):
        raise ValueError('Materialized quality ports cannot enter the packed resource frontier')
    return dict(variant=row['variant'], metric=metric, score=score, ci_low=low,
                ci_high=high, actual_bits=bits)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, folder, name):
    paths = []
    for extension in ('png', 'pdf', 'svg'):
        path = folder/(name+'.'+extension)
        if path.exists():
            raise FileExistsError('Refusing to replace a generated figure: '+str(path))
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {'Date': None} if extension == 'svg' else None
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.08, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def nondominated(points):
    # A displayed-set reference only, not a proof of a global Pareto frontier.
    ordered = sorted(points, key=lambda row: (row['actual_bits'], -row['score']))
    result, best = [], -math.inf
    for row in ordered:
        if row['score'] > best:
            result.append(row)
            best = row['score']
    return result


def doc_figure(cohorts, names, primary, styles, cap, folder, preview):
    allrows, plotted = [], []
    for title, data in cohorts:
        table = {row['variant']: row for row in data['table']}
        missing = set(names+['full8'])-table.keys()
        if missing:
            raise ValueError('Missing frozen Doc methods: '+str(sorted(missing)))
        points = [measured(table[name], 'anls') for name in names]
        full = measured(table['full8'], 'anls')
        plotted.append((title, data, points, full))
        allrows += [dict(cohort=title, **row) for row in points+[full]]
    width = 6.2 if preview else 11.6
    fig, axes = plt.subplots(1, len(plotted), figsize=(width, 5.6), sharey=True, squeeze=False)
    low = max(0, 100*min(row['ci_low'] for row in allrows)-1.4)
    high = min(100, 100*max(row['ci_high'] for row in allrows)+1.4)
    xmin = min(row['actual_bits'] for row in allrows if row['variant'] != 'full8')-.09
    xmax = max(row['actual_bits'] for row in allrows if row['variant'] != 'full8')+.22
    for index, (title, data, points, full) in enumerate(plotted):
        ax = axes[0, index]
        ax.axvspan(xmin, 1.8, color=PALETTE['green_1'], alpha=.55, zorder=0)
        ax.axvline(1.8, color=PALETTE['grey'], lw=1, ls=':', zorder=1)
        frontier = nondominated(points)
        ax.plot([r['actual_bits'] for r in frontier], [100*r['score'] for r in frontier],
                color=PALETTE['neutral'], lw=1.2, zorder=1)
        for row in points:
            label, color, marker = styles[row['variant']]
            score = 100*row['score']
            ax.errorbar(row['actual_bits'], score,
                        yerr=[[score-100*row['ci_low']], [100*row['ci_high']-score]],
                        fmt=marker, color=color, capsize=2.5, elinewidth=1.1,
                        ms=12 if row['variant'] == primary else 7,
                        markeredgewidth=.9, zorder=4 if row['variant'] == primary else 3)
        ax.axhline(100*full['score'], color=PALETTE['grey'], ls='--', lw=1.2)
        ax.text(.98, .96, f"Full reference: {full['actual_bits']:.2f} bit", transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color=PALETTE['dark'])
        p = next(r for r in points if r['variant'] == primary)
        native = next(r for r in points if r['variant'] == 'native4_8')
        ax.text(.98, .04, f"Primary - Native4: {100*(p['score']-native['score']):+.2f} pp",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
        ax.set(xlim=(xmin, xmax), ylim=(low, high), xlabel='Actual bits per KV scalar',
               title=f"{'a' if index == 0 else 'b'}  {title}\n{data['images']} documents / {data['questions_per_variant']} QA")
        ax.set_xticks([1.6, 1.8, 2.25, 3.0, 4.25])
        ax.set_xticklabels(['1.6', '1.8', '2.25', '3.0', '4.25'])
        ax.yaxis.grid(alpha=.12)
    axes[0, 0].set_ylabel('ANLS (%)')
    handles = [Line2D([], [], color=styles[name][1], marker=styles[name][2], ls='',
                      ms=10 if name == primary else 7, label=styles[name][0]) for name in names]
    handles.append(Line2D([], [], color=PALETTE['grey'], ls='--', lw=1.2, label='Full quality reference'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2 if preview else 4,
               columnspacing=1.4, handlelength=1.7)
    fig.text(.5, .005, f'Vision cap={cap} for all methods; green: <=1.8-bit cap. Error bars: 95% document bootstrap CI.',
             ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.29, top=.86, left=.12 if preview else .07, right=.97, wspace=.14)
    name = 'doc_development_preview' if preview else 'doc_resource_quality_two_cohorts'
    files = save(fig, folder, name)
    caption = ('Frozen methods shown on matched complete Doc cohorts. X includes actual visual payload, metadata, lookup, positions and charged encoder constants; the 1.8-bit limit is a cap, not the point coordinate. '
               'Full is a mean-quality reference at its stated >16-bit owner and is not plotted as a <=1.8-bit candidate. Error bars are marginal document-cluster bootstrap intervals, not paired difference intervals. '
               'The faint frontier is only the nondominated set among displayed frozen methods; no superiority over the strong gradient baseline is implied.')
    return files, allrows, caption


def text_figure(data, names, primary, styles, folder):
    table = {row['variant']: row for row in data['table']}
    order = ['full8', 'native4_8', 'base2_8', 'cold_meta_all_kv']+[name for name in names if name not in ('native4_8', 'base2_8', 'cold_meta_all_kv')]
    order = list(dict.fromkeys(order))
    if not set(order) <= table.keys():
        raise ValueError('Transfer table lacks a frozen displayed method')
    rows = [measured(table[name], 'accuracy') for name in order]
    fig, ax = plt.subplots(figsize=(9.1, 5.2))
    labels = []
    for position, row in enumerate(rows):
        label, color, marker = ('Full', PALETTE['grey'], 'o') if row['variant'] == 'full8' else styles[row['variant']]
        labels.append(label)
        score = 100*row['score']
        ax.errorbar(score, position,
                    xerr=[[score-100*row['ci_low']], [100*row['ci_high']-score]],
                    fmt=marker, color=color, capsize=3, elinewidth=1.4,
                    ms=12 if row['variant'] == primary else 7)
        ax.text(1.04, position, f"{row['actual_bits']:.4f}", transform=ax.get_yaxis_transform(),
                va='center', fontsize=12)
    ax.text(1.04, 1.03, 'Actual bit', transform=ax.transAxes, fontsize=11)
    ax.set(yticks=range(len(rows)), yticklabels=labels,
           xlim=(max(0, 100*min(r['ci_low'] for r in rows)-2), min(100, 100*max(r['ci_high'] for r in rows)+2)),
           xlabel='Official TextVQA soft accuracy (%)',
           title=f"TextVQA zero-shot transfer\n{data['images']} new images / {data['questions_per_variant']} QA")
    ax.invert_yaxis()
    ax.xaxis.grid(alpha=.12)
    fig.text(.52, .015, 'Frozen on Doc; no Text recalibration. Bars: 95% image-cluster bootstrap CI.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.37, right=.85, bottom=.15, top=.82)
    files = save(fig, folder, 'text_transfer_official_score')
    caption = ('New-image single-model transfer with Doc-frozen methods and no Text recalibration. Points are official ten-annotator MMF/TextVQA soft accuracy; horizontal intervals are image-cluster 95% bootstrap intervals. '
               'The separate right column reports each method actual complete bit cost, including Full; scores are not ANLS and do not establish multi-model generality.')
    return files, rows, caption


def mechanism_figure(data, folder):
    candidates = [
        ('full_k', 'Full K / V1', PALETTE['grey'], 'P', False),
        ('pq_fwht', 'PQ K / V1', PALETTE['blue_main'], 'o', False),
        ('query_r16', 'Query r16', PALETTE['teal'], '^', False),
        ('query_r16_norm', 'Query r16 + norm', PALETTE['teal'], '^', True),
        ('query_r32', 'Query r32', PALETTE['violet'], 's', False),
        ('query_r32_norm', 'Query r32 + norm', PALETTE['violet'], 's', True),
        ('query_r64', 'Query r64', PALETTE['dark'], 'D', False),
    ]
    lookup = {(row['candidate'], row['phase']): row for row in data['table'] if row['value_representation'] == 'v1'}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), sharex=True, sharey=True)
    rows = []
    for index, (phase, title) in enumerate([('question_prefill', 'Question prefill'), ('decode', 'Decode')]):
        for candidate, label, color, marker, opened in candidates:
            if (candidate, phase) not in lookup:
                raise ValueError('Missing fixed-query phase/candidate control')
            row = lookup[candidate, phase]
            x, y = float(row['score_centered_relative_mse']), float(row['output_relative_mse'])
            if not math.isfinite(x+y) or min(x, y) < 0:
                raise ValueError('Invalid local mechanism statistic')
            axes[index].plot(x, y, marker=marker, color=color, ms=8,
                             markerfacecolor='white' if opened else color, markeredgewidth=1.4, ls='')
            rows.append(dict(candidate=candidate, phase=phase, value_representation='v1',
                             score_relative_mse=x, output_relative_mse=y,
                             score_numerator='score_error_sse', score_normalizer='score_reference_centered_sse',
                             events=row['events'], head_count=row['head_count']))
        axes[index].set(title=f"{'a' if index == 0 else 'b'}  {title}",
                        xlabel='Visual score relative MSE\n(normalizer: centered Full scores)')
        axes[index].grid(alpha=.12)
    axes[0].set_ylabel(r'Post-$W_O$ output relative MSE')
    axes[0].set_xlim(-.015, max(row['score_relative_mse'] for row in rows)*1.08)
    axes[0].set_ylim(0, max(row['output_relative_mse'] for row in rows)*1.1)
    handles = [Line2D([], [], ls='', color=color, marker=marker, ms=8,
                      markerfacecolor='white' if opened else color, markeredgewidth=1.4, label=label)
               for _, label, color, marker, opened in candidates]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=4, fontsize=11)
    fig.text(.5, .005, 'Identical Full q / text; V1 fixed. Local counterfactuals only, not an ANLS prediction.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, bottom=.29, top=.84, wspace=.16)
    files = save(fig, folder, 'fixed_full_query_phase_mechanism')
    caption = ('Descriptive local counterfactuals on the fixed-query 16-document collection, with prefill and decode kept separate. '
               'Score numerator is all visual-score error SSE, including common offsets, normalized by centered Full-score SSE. '
               'Output error combines all query heads through W_O and includes the unchanged V1 readout error as well as K-induced effects. '
               'Current Full queries/text are identical across candidates; candidate states are not committed. Materialized projection references do not represent low-bit deployment owners. '
               'No sampling interval is invented from the aggregate table, and these local points are not predictions of final ANLS or student-trajectory error.')
    return files, rows, caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, nargs='+', required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--transfer', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path, default=ROOT/'audit/a2/round5_confirmation_v1.json')
    parser.add_argument('--fixed-query', type=Path, default=ROOT/'results/round5_v1/fixed_query/development16/summary.json')
    parser.add_argument('--preview-development', action='store_true')
    args = parser.parse_args()
    preview = args.preview_development
    if not preview and (args.confirmation is None or args.transfer is None):
        parser.error('Final figures require measured --confirmation and --transfer; use --preview-development for dev only')
    if preview and (args.confirmation is not None or args.transfer is not None):
        parser.error('Preview must not read confirmation or transfer inputs')
    folder = (args.output or ROOT/('figures/round5_preview' if preview else 'figures/round5_final')).resolve()
    if preview and not folder.is_relative_to((ROOT/'figures/round5_preview').resolve()):
        parser.error('Development previews must stay under figures/round5_preview')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Use an empty output directory to preserve generated evidence: '+str(folder))
    frozen = json.loads(args.freeze.read_text(encoding='utf-8-sig'))
    names, primary, secondary, finite, cap = frozen_methods(frozen)
    styles = descriptions(names, primary, secondary, finite)
    development = merge_development(args.development)
    cohorts = [('Development', development)]
    transfer = None
    if not preview:
        confirmation = load(args.confirmation)
        if confirmation.get('analysis_role') != 'confirmation':
            raise ValueError('Fresh Doc input must explicitly declare analysis_role=confirmation')
        if confirmation.get('samples_sha256') == development.get('samples_sha256'):
            raise ValueError('Confirmation reuses the development sample manifest')
        cohorts.append(('Fresh confirmation', confirmation))
        transfer = load(args.transfer)
        if transfer.get('primary_candidate') not in (None, primary):
            raise ValueError('Text summary primary differs from the pre-result freeze')
    fixed_query = load(args.fixed_query)
    # Validate the required methods/CIs before any output is written.
    for _, cohort in cohorts:
        table = {row['variant']: row for row in cohort['table']}
        for name in names+['full8']:
            measured(table[name], 'anls')
    if transfer is not None:
        table = {row['variant']: row for row in transfer['table']}
        for name in names+['full8']:
            measured(table[name], 'accuracy')
    chosen_font = publication_style()
    folder.mkdir(parents=True, exist_ok=True)
    files, data, captions = [], {}, {}
    produced, rows, caption = doc_figure(cohorts, names, primary, styles, cap, folder, preview)
    files += produced
    data['doc'] = rows
    captions['doc'] = caption
    write_csv(folder/'doc_figure_data.csv', rows)
    if transfer is not None:
        produced, rows, caption = text_figure(transfer, names, primary, styles, folder)
        files += produced
        data['text'] = rows
        captions['text'] = caption
        write_csv(folder/'text_figure_data.csv', rows)
    produced, rows, caption = mechanism_figure(fixed_query, folder)
    files += produced
    data['mechanism'] = rows
    captions['mechanism'] = caption
    write_csv(folder/'mechanism_figure_data.csv', rows)
    inputs = [*args.development, args.freeze, args.fixed_query]+([] if preview else [args.confirmation, args.transfer])
    manifest = dict(status='COMPLETE', preview=preview, font=chosen_font,
                    primary=primary, secondary_low_storage=secondary, original_finite_v=finite,
                    vision_cap=cap, inputs_sha256={str(path.resolve()): sha(path) for path in inputs},
                    script_sha256=sha(Path(__file__)),
                    outputs_sha256={path.name: sha(path) for path in files},
                    score_scope='Doc ANLS; Text official VQA soft accuracy; fixed-Full-q local MSE is separate',
                    uncertainty='95% source-document/image cluster bootstrap intervals from actual input tables; none fabricated',
                    resource_scope='actual complete owner including charged encoder constants; 1.8 is a cap',
                    frozen_selection='displayed method identities follow audit before new confirmation outcomes',
                    local_mechanism_scope=fixed_query['scope'])
    for name, value in [('figure_data.json', data), ('captions.json', captions), ('figure_manifest.json', manifest)]:
        with (folder/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
    print(json.dumps(dict(status='COMPLETE', preview=preview, figures=len(files)//3, directory=str(folder))))


if __name__ == '__main__':
    main()
