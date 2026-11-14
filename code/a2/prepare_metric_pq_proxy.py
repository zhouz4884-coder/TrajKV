"""Queue entrypoint for the independently versioned metric preparer."""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'prepare_metric_pq_round5.py'),run_name='__main__')
