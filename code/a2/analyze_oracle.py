"""Image-paired radial/directional intervention contrasts and measured energies."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from analyze import paired,write_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 scores=list(csv.DictReader((ROOT/'results/radial_oracle100_analysis/scores.csv').open(encoding='utf-8-sig')))
 groups=defaultdict(dict)
 for r in scores:groups[r['variant']][(r['image_id'],1)]=float(r['anls'])
 contrasts=[]
 for enc in ('mad','lloyd'):
  for left,right,label in [(enc+'_A10',enc+'_A00','restore_norm_at_quant_direction'),(enc+'_A01',enc+'_A00','restore_direction_at_quant_norm'),
    ('A11_fp16',enc+'_A10','restore_direction_at_true_norm'),('A11_fp16',enc+'_A01','restore_norm_at_true_direction'),
    (enc+'_A01',enc+'_A10','direction_arm_minus_norm_arm')]:
   contrasts.append({'encoder':enc,'contrast':label,'left':left,'right':right,**paired(groups[left],groups[right])})
  a00=groups[enc+'_A00'];a10=groups[enc+'_A10'];a01=groups[enc+'_A01'];a11=groups['A11_fp16']
  interaction={k:a11[k]-a10[k]-a01[k]+a00[k] for k in a00}
  contrasts.append({'encoder':enc,'contrast':'factorial_interaction','left':'A11-A10-A01+A00','right':'zero',**paired(interaction,{k:0. for k in a00})})
 stat=defaultdict(list)
 for run in a.runs:
  for f in (ROOT/'results'/run).rglob('records.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line)
    if r['variant'] not in ('mad_A00','lloyd_A00'):continue
    enc=r['oracle_encoder'];zs=r['radial_angular_statistics'];rad=sum(x['radial_squared_error'] for x in zs);ang=sum(x['angular_squared_error'] for x in zs)
    total=sum(x['total_squared_error'] for x in zs)
    stat[enc].append({'id':r['session_id'],'radial_energy_share':rad/(rad+ang),'decomposition_relative_error':abs(total-rad-ang)/max(total,1e-20),
     'mean_cosine':np.mean([x['mean_cosine'] for x in zs]),'true_norm_cv':np.mean([x['true_norm_cv'] for x in zs]),
     'quant_norm_cv':np.mean([x['quant_norm_cv'] for x in zs]),'within_norm_corr':np.mean([x['norm_correlation_within_group'] for x in zs if x['norm_correlation_within_group'] is not None]),
     'undefined_corr_count':sum(x['norm_correlation_within_group'] is None for x in zs)})
 table=[{'encoder':enc,'images':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'}} for enc,rows in stat.items()]
 write_csv(a.output/'paired_contrasts.csv',contrasts);write_csv(a.output/'energy_summary.csv',table)
 result={'contrasts':contrasts,'energies':table,'scope':'all arms materialized FP16 diagnostics; conditional factorial contrasts, not causal attribution percentages'}
 (a.output/'mechanism.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
