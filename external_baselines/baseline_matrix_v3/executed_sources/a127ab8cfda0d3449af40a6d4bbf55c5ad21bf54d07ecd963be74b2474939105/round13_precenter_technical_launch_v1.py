"""Bounded synthetic CUDA QA0 alongside instrumented quality inference.

No neural model is loaded. A separate process allocator is capped at 10% of
one card; overlap is recorded and is not a controlled speed benchmark.
"""
from pathlib import Path
import runpy
import torch
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(.10,0)
print('ROUND13_QA0_SYNTHETIC_ONLY_ALLOCATOR_CAP_10PCT_QUALITY_TIMING_OVERLAP',flush=True)
runpy.run_path(str(Path(__file__).with_name('round13_precenter_technical_v1.py')),run_name='__main__')
