"""Final read-only source capture including the four early generic Mean jobs.

The preserved collector is adapted in memory. Generic-launcher inventories
are explicitly distinguished from the actual working-tree entry they ran.
No worker, original receipt, frozen source or prior capture is modified.
"""
import collect_round8_runtime_sources as original


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Preserved collector adapter anchor changed')
    return source.replace(before, after, 1)


def adapted_scan():
    scan = original.REMOTE_SCAN
    scan = replace_once(scan,
        "jobs=[]; gaps=[]; snapshot_root=root/'jobs'\nfor snapshot in sorted(snapshot_root.glob('*.source')):",
        """jobs=[]; gaps=[]; stage_root=root; project_root=root.parents[1]
registered=[(snapshot,stage_root) for snapshot in sorted((stage_root/'jobs').glob('*.source'))]
for generic_name in ('round8_mean_select2_v1_doc','round8_mean_select2_v1_text',
                     'round8_mean_select2_v2_doc','round8_mean_select2_v2_text'):
    generic=project_root/'jobs'/(generic_name+'.source')
    if generic.is_dir():registered.append((generic,project_root))
for snapshot,job_root in registered:
    root=job_root; snapshot_root=root/'jobs'""")
    scan = replace_once(scan,
        "'complete_final_ray.json','complete_mean.json','manifest.json'):",
        """'complete_final_ray.json','complete_mean.json','complete_postcal.json',
                'complete_postcal_train.json','complete_final_measurement.json',
                'complete_backend_diagnostic.json','complete_backend_timing.json','manifest.json'):""")
    scan = replace_once(scan,
        "source_entries=manifest.get('sources',{})\n        if not isinstance(source_entries,dict): continue  # native dump: sources is an integer count",
        """source_entries=manifest.get('sources',{})
        source_entries=dict(source_entries) if isinstance(source_entries,dict) else {}
        for source_key in ('source_sha256','postcal_source_sha256','mean_source_sha256',
                           'final_ray_source_sha256','postres_source_sha256'):
            extra=manifest.get(source_key,{})
            if not isinstance(extra,dict):continue
            for key,value in extra.items():
                if not key.endswith('.py') or not isinstance(value,str):continue
                normalized=key if key.startswith('code/') else 'code/'+key if key.startswith('experiments/') else 'code/a2/'+key
                if normalized in source_entries and source_entries[normalized]!=value:raise ValueError('Conflicting consumed-source identities')
                source_entries[normalized]=value""")
    scan = replace_once(scan,
        "record=dict(job=name,source_snapshot=str(snapshot),source_inventory_sha256=sha(inventory_raw),",
        """executed=launch.get('argv_executed') or job.get('argv',[])
    entry=Path(executed[1]) if len(executed)>1 else None
    entry_in_snapshot=bool(entry and str(entry).startswith(str(snapshot)+'/'))
    entry_current_sha=source(entry) if entry and entry.is_file() else None
    record=dict(job=name,source_snapshot=str(snapshot),source_inventory_sha256=sha(inventory_raw),
        actual_entry_path=str(entry) if entry else None,actual_entry_in_snapshot=entry_in_snapshot,
        observed_entry_file_sha256=entry_current_sha,
        entry_evidence_scope=('Popen actual argv plus immutable executed entry/source inventory' if entry_in_snapshot else
            'Popen actual argv targets SR working tree; global launcher snapshot is availability only; observed current entry hash is not by itself proof of historical execution'),""")
    return scan


if __name__ == '__main__':
    original.REMOTE_SCAN = adapted_scan()
    original.main()
