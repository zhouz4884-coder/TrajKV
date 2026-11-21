"""Use unchanged exporter_v3; exact Root-approved collection exclusions only."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path('${PROJECT_ROOT}')
WORK=Path('${EVIDENCE_ROOT}/results/publication/OnlyTerp_Hmix23_12_complete_CPU_20261121_v1')
sys.path.insert(0,str(ROOT/'ops'))
import export_public_v3 as exporter
from public_phase_contract_v2 import sha,require
scope=json.loads((WORK/'SOURCE_COLLECTION_SCOPE_V2.json').read_text(encoding='utf-8'))
require(sha(ROOT/'ops/export_public_v3.py')==scope['old_exporter_source_sha256'],'Frozen exporter changed')
excluded={(ROOT/name).resolve() for name in scope['excluded_exact_project_paths']}
original_collect=exporter.collect
def scoped_collect(*args,**kwargs):
    selected,duplicates,artifacts,closure,matrix=original_collect(*args,**kwargs)
    for name in list(selected):
        if selected[name]['path'].resolve() in excluded:del selected[name]
    return selected,duplicates,artifacts,closure,matrix
exporter.collect=scoped_collect
plan=json.loads((WORK/'EXPORT_PLAN_V2.json').read_text(encoding='utf-8'))
sys.argv=['export_public_v3.py','--stage',plan['stage'],'--tag',plan['tag'],'--contract',plan['primary_contract'],'--support-file-lists',*plan['support_file_lists'],'--require-complete']
exporter.main()
