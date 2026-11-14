"""Separate reader for actual mixed base widths and 1-bit K residuals."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
s=(HERE/'fused_masked.py').read_text(encoding='utf-8')
s=s.replace('BITS: tl.constexpr, REFINE_K:', 'BITS: tl.constexpr, VBITS: tl.constexpr, KRBITS: tl.constexpr, REFINE_K:')
start=s.index('                vc=');end=s.index('                use_residual',start)
old=s[start:end];new=old.replace('BITS','VBITS');s=s[:start]+new+s[end:]
s=s.replace('                        b=tl.load(RK+total+ix//8,valid[None,:],0)\n                        code=((a>>(ix%8))&1)+2*((b>>(ix%8))&1)',
 '''                        code=(a>>(ix%8))&1
                        if KRBITS == 2:
                            b=tl.load(RK+total+ix//8,valid[None,:],0)
                            code=code+2*((b>>(ix%8))&1)''')
assert 'b=tl.load(RK+total' in s
s=s.replace('refinement_mask=None, page_size=256):','refinement_mask=None, page_size=256, bits_v=None, kr_bits=2):')
s=s.replace('    assert b==1 and bits in (1,2,4)','    bits_v=bits if bits_v is None else bits_v\n    assert b==1 and bits in (1,2,4) and bits_v in (1,2,4) and kr_bits in (1,2)')
s=s.replace('numel(), bits, active_k, active_v, observe,','numel(), bits, bits_v, kr_bits, active_k, active_v, observe,')
(HERE/'fused_rates.py').write_text(s,encoding='utf-8')
print(json.dumps({'source':'fused_masked.py','derived':'fused_rates.py','sha256':hashlib.sha256(s.encode()).hexdigest()}))
