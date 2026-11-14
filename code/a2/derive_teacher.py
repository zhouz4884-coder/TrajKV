"""Diagnostic-only forced-reference executor; logits remain in memory, never scored as free answers."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'session_boundary.py').read_text(encoding='utf-8')
needle='owned_initial=bank.owned();turns=[];snapshots=[];probe_rows=[];previous=None'
assert needle in s;s=s.replace(needle,needle+';teacher_logits=[]')
needle='            margin=margins(logits);token_margins.append(margin)'
assert needle in s;s=s.replace(needle,"            if target is not None:teacher_logits.append(logits.float().cpu().clone())\n"+needle)
s=s.replace('return result,snapshots','return result,teacher_logits')
(HERE/'session_teacher.py').write_text('# Forced-reference diagnostic proxy; excluded from task accuracy.\n'+s,encoding='utf-8')
print('Created GPU teacher-forced likelihood diagnostic executor')
