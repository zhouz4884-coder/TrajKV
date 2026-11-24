"""Portable admission of reused historical producers and derived views.

These checks preserve the original retrieval contract; they do not assert a
fresh inference run or replay private tensors. No network or authentication.
"""
from __future__ import annotations
from pathlib import PurePosixPath


def require(value, message):
    if not value:
        raise ValueError('ROUND12_LEGACY_PUBLIC: ' + message)


def bundle(source, spec):
    files = spec['files']
    require({'manifest.json','records.jsonl','complete.json','source_inventory.json','launch_receipt.json'}<=set(files),
            'original producer core member omitted')
    proof = source.doc(spec['source_proof'])
    require(proof.get('remote_hashes_computed_independently') is True
            and proof.get('status') in ('PASS_REMOTE_SOURCE_SHA256_EXACT',
                'PASS_R7_EXECUTED_DRIVER_AND_DEPENDENCY_SOURCE_SHA',
                'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'),
            'original independent source proof missing')
    for item in proof['files']:
        require(item['file'] in files, 'original source proof member omitted: ' + item['file'])
        source.original(files[item['file']], item['sha256'])
    raw_proof = source.doc(spec['raw_receipt'])
    require(raw_proof.get('remote_hashes_computed_independently') is True
            and raw_proof.get('status') in ('PASS_REMOTE_RAW_SHA256_EXACT',
                'PASS_CONFIRMATION_EXECUTED_SOURCE_AND_RAW_SHA'),
            'original independent raw readback missing')
    for item in raw_proof['files']:
        if item['file'] in ('manifest.json', 'records.jsonl', 'complete.json'):
            source.original(files[item['file']], item['sha256'])
    manifest, done = (source.doc(files[x]) for x in ('manifest.json', 'complete.json'))
    source.original(files['manifest.json'], done['manifest_sha256'])
    source.original(files['records.jsonl'], done['records_jsonl_sha256'])
    require(done.get('coverage_complete') is True and done.get('quality_scored') is False
            and manifest.get('references_read') is False
            and manifest.get('prediction_cache_used') is False,
            'historical source is not a closed reference-free inference producer')
    inventory = source.doc(files['source_inventory.json'])
    launch = source.doc(files['launch_receipt.json'])
    source.original(files['source_inventory.json'], launch['source_inventory_sha256'])
    if 'source_job.json' in files:
        source.original(files['source_job.json'], launch['job_sha256'])
    available = {source.files[rel]['source_sha256'] for rel in files.values()}
    for logical, digest in manifest.get('source_hashes', {}).items():
        require(inventory.get(logical) == digest, 'historical source inventory differs')
        # Older native external ports archived the entry source and full hash
        # inventory, while R7/R11 archived dependency bodies too. Preserve that
        # declared evidence scope instead of fabricating missing old bodies.
        if proof['status'] != 'PASS_REMOTE_SOURCE_SHA256_EXACT':
            require(digest in available, 'historical executed dependency body omitted: ' + logical)
    for digest in done.get('executed_legacy_source_sha256', {}).values():
        require(digest in available, 'historical legacy dependency omitted')
    rows = source.lines(files['records.jsonl'])
    require([r['id'] for r, _, _ in rows] == manifest['expected_ids']
            and len(rows) == done['recorded_QA'] == done['expected_QA'],
            'historical physical producer coverage differs')
    return manifest, done, rows


def binding(cell, source, manifest, complete):
    p = cell['producer']
    require(cell['kind'] == 'reused_reference', 'legacy binding cannot admit new inference')
    if p['kind'] == 'legacy_source_proof':
        original, done, _ = bundle(source, p)
        require(original == manifest and done == complete, 'legacy producer anchor differs')
        return
    require(p['kind'] == 'legacy_derived_view'
            and manifest.get('schema') == 'baseline_matrix.derived_prediction_view.v1'
            and manifest.get('inference_run') is False
            and complete.get('derived_prediction_view') is True,
            'derived view must be explicitly historical, not another inference')
    source.original(p['provenance'], complete['provenance_jsonl_sha256'])
    lineage = source.lines(p['provenance'])
    view = source.lines(cell['paths']['records'])
    require(len(lineage) == len(view), 'derived provenance coverage differs')
    sources = {}
    for logical, spec in p['sources'].items():
        m, d, rows = bundle(source, spec)
        sources[logical] = (spec, m, d, rows)
    for (row, _, _), (link, _, _) in zip(view, lineage):
        require(row['id'] == link['id'] and link['source_run'] in sources,
                'derived source identity differs')
        spec, _, _, rows = sources[link['source_run']]
        for name, digest in link['source_raw_hashes'].items():
            source.original(spec['files'][name], digest)
        n = link['source_line_number']
        require(1 <= n <= len(rows), 'derived source row out of range')
        original, number, digest = rows[n - 1]
        require(number == n and digest == link['source_exact_line_sha256']
                and original == row, 'derived row is not exact original producer content')

