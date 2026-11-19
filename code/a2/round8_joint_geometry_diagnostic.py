"""CPU-only geometry of existing K joint-gain interventions on native samples.

The original all-image frame and sampled native source are reused unchanged.
Only the 23 selected first-stage K heads are refitted with the same production
joint_refit_key function. Other heads have exactly zero intervention delta.
No model/labels/GPU/fit-book work is performed and no tensor is written.
Radial/angular reconstruction terms describe geometry, not task causality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from round8_joint_gain import joint_refit_key
from round6_depth_codec import decode_sign_stage


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def hard_decode(template, gain, sign_code=None, alpha=None):
    base = (template.float()*gain.float()).half()
    return base if sign_code is None else decode_sign_stage(base, sign_code, alpha)


def geometry(source, decoded):
    """Float64 exact-tensor geometry, including zero-norm corner cases."""
    z, y = source.double(), decoded.double()
    require(z.ndim == 2 and z.shape[-1] == 128 and z.shape == y.shape
        and bool(torch.isfinite(z).all()) and bool(torch.isfinite(y).all()), 'Finite equal CPU D128 rows required')
    radius, amplitude = z.square().sum(-1).sqrt(), y.square().sum(-1).sqrt()
    dot = (z*y).sum(-1)
    product = radius*amplitude
    defined = product > 0
    cosine = torch.where(defined, dot/product.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(dot))
    radial = (radius-amplitude).square()
    # This is 2*r*a*(1-cos), evaluated without cancellation from 1-cos.
    angular = 2*(product-dot)
    sse = (z-y).square().sum(-1)
    error = sse-radial-angular
    require(bool((error.abs() <= 1e-8+1e-11*sse).all()), 'Float64 SSE radial/angular identity failed')
    cosine_form = torch.where(defined, 2*product*(1-cosine), torch.zeros_like(dot))
    require(bool(((cosine_form-angular).abs() <= 1e-8+1e-11*sse).all()), 'Angular cosine formula failed')
    return dict(sse=sse, source_norm=radius, reconstruction_norm=amplitude, cosine=cosine,
        cosine_defined=defined, radial=radial, angular=angular, identity_abs_error=error.abs())


class Accumulator:
    def __init__(self):
        self.rows = self.improved = self.changed = self.cos_improved = self.cos_worsened = 0
        self.radial_improved = self.angular_improved = self.both_improved = 0
        self.sse_improved_cos_worsened = self.sse_improved_angular_worsened = 0
        self.sums = {side: {key: 0. for key in ('sse','source_norm','reconstruction_norm','cosine','radial','angular')}
                     for side in ('before','after')}
        self.cos_counts = dict(before=0, after=0)
        self.identity_error = 0.
        self.delta = []

    def add(self, before, after, changed):
        n = len(before['sse']); delta = after['sse']-before['sse']
        require(bool((delta <= 0).all()), 'Joint refit increased an actual FP16 row SSE')
        self.rows += n; self.improved += int((delta < 0).sum()); self.changed += int(changed.sum())
        defined = before['cosine_defined'] & after['cosine_defined']
        cd = after['cosine']-before['cosine']
        cos_up, cos_down = defined & (cd > 1e-12), defined & (cd < -1e-12)
        rd, ad = after['radial']-before['radial'], after['angular']-before['angular']
        rt, at = 1e-12*(1+before['radial'].abs()), 1e-12*(1+before['angular'].abs())
        self.cos_improved += int(cos_up.sum()); self.cos_worsened += int(cos_down.sum())
        self.radial_improved += int((rd < -rt).sum()); self.angular_improved += int((ad < -at).sum())
        self.both_improved += int(((rd < -rt) & (ad < -at)).sum())
        self.sse_improved_cos_worsened += int(((delta < 0) & cos_down).sum())
        self.sse_improved_angular_worsened += int(((delta < 0) & (ad > at)).sum())
        self.delta.append(delta.clone())
        for side, value in (('before',before),('after',after)):
            self.identity_error = max(self.identity_error,float(value['identity_abs_error'].max()))
            self.cos_counts[side] += int(value['cosine_defined'].sum())
            for key in self.sums[side]:
                tensor = value[key]
                if key == 'cosine':
                    tensor = tensor[value['cosine_defined']]
                self.sums[side][key] += float(tensor.sum())

    def row(self):
        require(self.rows > 0, 'Empty geometry scope')
        row = dict(rows=self.rows, sse_improved_rows=self.improved, fields_changed_rows=self.changed,
            cosine_improved_rows=self.cos_improved, cosine_worsened_rows=self.cos_worsened,
            radial_component_improved_rows=self.radial_improved, angular_component_improved_rows=self.angular_improved,
            both_components_improved_rows=self.both_improved,
            sse_improved_but_cosine_worsened_rows=self.sse_improved_cos_worsened,
            sse_improved_but_angular_component_worsened_rows=self.sse_improved_angular_worsened,
            maximum_float64_identity_abs_error=self.identity_error)
        for side in ('before','after'):
            row['cosine_defined_rows_'+side] = self.cos_counts[side]
            for key, value in self.sums[side].items():
                denominator = self.cos_counts[side] if key == 'cosine' else self.rows
                row[key+'_mean_'+side] = value/denominator if denominator else None
            row['mse_per_scalar_'+side] = self.sums[side]['sse']/self.rows/128
        for key in ('sse','source_norm','reconstruction_norm','radial','angular'):
            row[key+'_mean_delta'] = row[key+'_mean_after']-row[key+'_mean_before']
        row['cosine_mean_delta'] = (row['cosine_mean_after']-row['cosine_mean_before']
            if row['cosine_mean_before'] is not None and row['cosine_mean_after'] is not None else None)
        delta = torch.cat(self.delta)
        for q, label in ((0.,'min'),(.05,'q05'),(.25,'q25'),(.5,'median'),(.75,'q75'),(.95,'q95'),(1.,'max')):
            row['row_sse_delta_'+label] = float(torch.quantile(delta,q))
        return row


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def cpu_toy():
    radial = geometry(torch.ones(1,128),torch.full((1,128),.5).half())
    require(float(radial['angular'].abs().max()) < 1e-10 and float(radial['radial']) > 0, 'Pure radial toy failed')
    source = torch.zeros(1,128); source[0,0] = 1
    other = torch.zeros(1,128,dtype=torch.float16); other[0,1] = 1
    angular = geometry(source,other)
    require(float(angular['radial']) == 0 and float(angular['angular']) == 2, 'Pure angular toy failed')
    zero = geometry(torch.zeros(1,128),torch.zeros(1,128,dtype=torch.float16))
    require(not bool(zero['cosine_defined'].any()) and float(zero['sse']) == 0, 'Zero-norm convention failed')
    z = torch.tensor([2.,1.]).repeat(64)[None]
    t = torch.tensor([1.,3.]).repeat(64)[None]
    g = (z.norm(dim=-1,keepdim=True)/t.norm(dim=-1,keepdim=True)).half()
    base = (t*g.float()).half(); residual = z-base.float()
    sign = (residual >= 0).long().reshape(1,4,32)
    code = (sign << torch.arange(32)).sum(-1).int()
    alpha = residual.abs().mean(-1).half()
    ng,na,stats = joint_refit_key(z,t,g,code,alpha)
    before,after = geometry(z,hard_decode(t,g,code,alpha)),geometry(z,hard_decode(t,ng,code,na))
    require(float(ng) == .75 and float(na) == 1.25 and float(after['sse']) == 0, 'Production joint-refit interface toy failed')
    accumulator = Accumulator(); accumulator.add(before,after,(ng[:,0]!=g[:,0]) | (na!=alpha))
    return dict(status='PASS',scope='CPU toy decomposition/interface only; not a source/model/task result',
        pure_radial=True,pure_angular=True,zero_norm_handled=True,expected_gain=.75,expected_alpha=1.25,
        joint_refit_stats=stats,toy_geometry=accumulator.row())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump',type=Path); p.add_argument('--splits',type=Path); p.add_argument('--old-book',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--toy-only',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(os.environ.get('CUDA_VISIBLE_DEVICES','') == '', 'This diagnostic requires CUDA_VISIBLE_DEVICES empty')
    require(not args.output.exists(),'Use a new diagnostic output; frozen results are not overwritten')
    args.output.mkdir(parents=True)
    toy = cpu_toy()
    (args.output/'toy_check.json').write_text(json.dumps(toy,indent=2,allow_nan=False),encoding='utf-8')
    if args.toy_only:
        (args.output/'complete.json').write_text(json.dumps(dict(status='PASS',CPU_toy_only=True)),encoding='utf-8')
        print(json.dumps(dict(status='PASS',CPU_toy_only=True,threads=4))); return
    require(all((args.dump,args.splits,args.old_book)),'Native dump, frozen split and old book are required')
    manifest,split = load(args.dump/'manifest.json'),load(args.splits)
    require(manifest['schema'] == 'trajkv.round8.native-dump.v1' and manifest['old_book_sha256'] == sha(args.old_book),'Native dump/old encoder mismatch')
    book = torch.tensor(load(args.old_book)['centers'],dtype=torch.float16)
    require(tuple(book.shape) == (16,4),'Original K1 G4_16 book required')
    membership = {}
    for group in ('train','select'):
        require(len(split[group]) == 32 and len(set(split[group])) == 32,'Frozen 32-source memberships required')
        for source in split[group]:
            require(source not in membership,'Train/select source overlap'); membership[source]=group
    require(len(manifest['entries']) == 64 and {x['source_id'] for x in manifest['entries']} == set(membership),'Native source union differs from frozen train32/select32')
    stages = [tuple(x) for x in manifest['selected_stages']]
    protected = {(layer,head) for axis,layer,head,stage in stages if axis == 'k' and stage == 0}
    require(len(stages) == 43 and len(protected) == 23 and all(x[3] == 0 for x in stages),'Frozen 23K20V first-stage mask required')
    head_rows,source_rows,source_receipts = [],[],[]
    pooled = {(part,scope):Accumulator() for part in ('train','select') for scope in ('protected23','all112')}
    for number,entry in enumerate(manifest['entries']):
        path = args.dump/entry['file']; require(sha(path) == entry['sha256'],'Native source shard changed')
        shard = torch.load(path,map_location='cpu',weights_only=False)
        source_id,part = entry['source_id'],membership[entry['source_id']]
        require(shard['source_id'] == source_id and shard.get('diagnostic_source_owner') is True
            and shard.get('deployable') is False and shard['selected_stages'] == manifest['selected_stages'],'Native source identity/frame/mask differs')
        require(shard['layers']*shard['heads'] == 112 and len(shard['units']) == 112,'Current diagnostic scope is the primary 112-head model')
        seen = set(); per_source = {scope:Accumulator() for scope in ('protected23','all112')}
        kept = len(shard['sampled_token_indices'])
        for unit in shard['units']:
            unit_id = int(unit['layer']),int(unit['head']); require(unit_id not in seen,'Duplicate native head'); seen.add(unit_id)
            z,gain = unit['k_source'].float(),unit['k_gain'].half()
            require(len(z) == kept and tuple(z.shape) == (kept,128),'Unequal native token sample across heads')
            template = book.float()[unit['k_index4'].long()].flatten(-2)
            base = hard_decode(template,gain)
            require(torch.equal(base,unit['k_old_base']),'Original native hard base identity failed')
            selected = unit_id in protected
            require(len(unit['k_residual_codes']) == len(unit['k_residual_alphas']) == int(selected),'Native old residual mask does not match protected K heads')
            if selected:
                code,alpha = unit['k_residual_codes'][0],unit['k_residual_alphas'][0]
                new_gain,new_alpha,stats = joint_refit_key(z,template,gain,code,alpha)
                before_decode,after_decode = hard_decode(template,gain,code,alpha),hard_decode(template,new_gain,code,new_alpha)
                changed = (gain[:,0]!=new_gain[:,0]) | (alpha!=new_alpha)
            else:
                before_decode = after_decode = base
                changed = torch.zeros(kept,dtype=torch.bool)
            before,after = geometry(z,before_decode),geometry(z,after_decode)
            if selected:
                require(stats['improved_rows'] == int((after['sse'] < before['sse']).sum())
                    and math.isclose(stats['source_sse_before'],float(before['sse'].sum()),rel_tol=1e-12,abs_tol=1e-10)
                    and math.isclose(stats['source_sse_after'],float(after['sse'].sum()),rel_tol=1e-12,abs_tol=1e-10),
                    'Shared joint-refit statistics differ from independently decoded geometry')
            head = Accumulator(); head.add(before,after,changed)
            for scope in (('protected23','all112') if selected else ('all112',)):
                per_source[scope].add(before,after,changed); pooled[part,scope].add(before,after,changed)
            head_rows.append(dict(source_index=number,source_id=source_id,partition=part,
                layer=unit_id[0],head=unit_id[1],protected=selected,**head.row()))
        for scope,accumulator in per_source.items():
            source_rows.append(dict(source_index=number,source_id=source_id,partition=part,scope=scope,**accumulator.row()))
        protected_mean,all_mean = per_source['protected23'].row(),per_source['all112'].row()
        require(math.isclose(all_mean['sse_mean_delta'],protected_mean['sse_mean_delta']*23/112,rel_tol=1e-10,abs_tol=1e-9),'Protected/all-head dilution identity failed')
        source_receipts.append(dict(source_id=source_id,partition=part,shard_sha256=entry['sha256'],sampled_tokens=kept,
            protected_heads=23,total_heads=112,unprotected_intervention_delta_exactly_zero=True))
        print(json.dumps(dict(source_index=number,partition=part,protected_rows=protected_mean['rows'],
            improved_rows=protected_mean['sse_improved_rows'],protected_mean_sse_delta=protected_mean['sse_mean_delta'])),flush=True)
        del shard
    aggregates = []
    for part in ('train','select'):
        for scope in ('protected23','all112'):
            rows = [row for row in source_rows if row['partition'] == part and row['scope'] == scope]
            require(len(rows) == 32,'Missing source-balanced group')
            average_keys = [key for key in rows[0] if '_mean_' in key or key.startswith('mse_per_scalar_')]
            aggregate = dict(partition=part,scope=scope,sources=32,source_weight=1/32,
                source_balance='equal weight per source; equal native head/token row weight within each source',
                row_counts_and_quantiles_scope='pooled rows; distinct from equal-source means')
            aggregate['source_balanced_means'] = {key:sum(row[key] for row in rows if row[key] is not None)/sum(row[key] is not None for row in rows)
                if any(row[key] is not None for row in rows) else None for key in average_keys}
            aggregate['cosine_defined_sources_before'] = sum(row['cosine_mean_before'] is not None for row in rows)
            aggregate['cosine_defined_sources_after'] = sum(row['cosine_mean_after'] is not None for row in rows)
            aggregate['row_statistics'] = pooled[part,scope].row()
            aggregates.append(aggregate)
    csv_write(args.output/'head_metrics.csv',head_rows)
    csv_write(args.output/'source_metrics.csv',source_rows)
    summary = dict(schema='trajkv.round8.joint-gain-native-geometry.v1',status='COMPLETE',threads=4,GPU_used=False,
        dump_manifest_sha256=sha(args.dump/'manifest.json'),source_split_sha256=sha(args.splits),old_book_sha256=sha(args.old_book),
        source_script_sha256=sha(__file__),joint_refit_script_sha256=sha(Path(__file__).with_name('round8_joint_gain.py')),
        selected_stages=manifest['selected_stages'],sources=source_receipts,groups=aggregates,
        decoder='FP16(FP16(final_effective_gain*original_template)+alpha*fixed_original_sign)',
        decomposition='float64 SSE=(source_norm-reconstruction_norm)^2+2*source_norm*reconstruction_norm*(1-cosine)',
        decomposition_numeric_tolerance=dict(absolute=1e-8,relative_to_row_SSE=1e-11),
        zero_norm_convention='cosine undefined and excluded from cosine means; angular contribution is zero when either norm is zero',
        row_statistics='every sampled row scored; counts and exact delta quantiles saved rather than a large per-row table; no tensors written',
        interpretation='radial/angular are reconstruction geometry; angular term depends on reconstruction norm as well as cosine; neither component is a causal task contribution',
        unchanged='source/frame/book/J/sign/mask and all unprotected heads; only pre-existing protected gain/alpha slots may change',
        scope='native sampled-coordinate diagnostic; no labels/model/task-quality/throughput/serving-speed result')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    files = {name:sha(args.output/name) for name in ('summary.json','head_metrics.csv','source_metrics.csv','toy_check.json')}
    (args.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',schema=summary['schema'],sources=64,
        files=files,GPU_used=False,task_quality_eligible=False,serving_timing_eligible=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',sources=64,head_rows=len(head_rows),source_scope_rows=len(source_rows),GPU_used=False)),flush=True)


if __name__ == '__main__':
    main()
