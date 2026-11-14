"""Scientific figures from complete frozen64 evidence only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'figures/a2_confirmation'
C={'blue':'#0F4D92','green':'#8BCF8B','red':'#B64342','teal':'#42949E','violet':'#9A4D8E','grey':'#CFCECE'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.dpi':300})
def save(fig,name):
 OUT.mkdir(exist_ok=True,parents=True)
 for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 plt.close(fig)
def error(ax,x,r,color,marker='o',label=None):
 y=100*r['anls'];ax.errorbar(x,y,yerr=[[max(0,y-100*r['ci_low'])],[max(0,100*r['ci_high']-y)]],fmt=marker,color=color,capsize=2.3,markersize=5,label=label)
def main():
 folder=ROOT/'results/frozen_confirm64_analysis';data=json.loads((folder/'summary.json').read_text());assert data['sessions']==2112 and data['unique_images']==64
 table={r['variant']:r for r in data['table']};captions={}
 fig,ax=plt.subplots(figsize=(7.4,4.0),layout='constrained')
 selected=[('calib1_mean_fp32','Centered K1, FP32 offset',C['violet'],'v'),('lloyd_joint','Binary Lloyd',C['grey'],'o'),
  ('lloyd_rms125_joint','Midpoint + RMS',C['green'],'s'),('calib_k2v1_fp32','Calib K2/V1 + FP32',C['violet'],'^'),
  ('k2_global_mad_v1','K2 / global V1',C['teal'],'D'),('direction_vq2_v1','Direction code',C['teal'],'s'),
  ('k2_lloyd_v1','K2 / Lloyd V1',C['red'],'o'),('storage_primary','State-matched calibration',C['blue'],'*'),
  ('calib2_fp32','Calib K2/V2 + FP32',C['violet'],'x'),('base2_8','Native K2/V2',C['red'],'s'),('native4_8','Native4',C['red'],'^')]
 for name,label,color,marker in selected:error(ax,table[name]['effective_visual_bits'],table[name],color,marker,label)
 ax.axhline(100*table['full8']['anls'],color='#555555',ls='--',lw=1,label='FP16 quality reference');ax.axvline(1.8,color='#888888',ls=':',lw=1)
 ax.set(xlabel='Actual visual-cache bits / KV scalar',ylabel='ANLS (%)',xlim=(.9,4.45),ylim=(0,100),title='Frozen64: observed storage–quality trade-off')
 ax.legend(fontsize=7,ncol=2,frameon=False,loc='lower right');ax.yaxis.grid(alpha=.12);ax.set_axisbelow(True);save(fig,'confirm64_storage_quality')
 captions['confirm64_storage_quality']='64 current-cycle image-disjoint documents,286 questions, same cap8 inputs. Actual tensor-owner bytes include metadata, gains, positions and required decoder constants. Encoder-only calibration parameters are listed separately and conservatively charged for budget eligibility. Full FP16 is a quality reference, not a mathematical upper bound. KVarN materialized quality-port storage is not plotted as a low-bit runtime. Error bars:95% whole-image bootstrap.'
 fig,ax=plt.subplots(figsize=(7.0,3.8),layout='constrained');allrow=table['read_all']
 labels={'read_none':'No refinement','read_all':'All refinement','read_periodic':'Periodic','read_prefill_only':'Question prefill','read_random':'Random','read_mass':'Mass','read_risk':'Risk proxy','read_thumb_random':'Thumbnail priority'}
 label_positions={'read_none':(-2,79.6),'read_all':(82,93.6),'read_prefill_only':(4,94.1),'read_risk':(41,92.9),
  'read_mass':(43,90.3),'read_thumb_random':(43,85.4),'read_random':(43,82.5),'read_periodic':(27,79.8)}
 for name,label in labels.items():
  row=table[name];x=100*row['extra_reads_total']/allrow['extra_reads_total'];color=C['blue'] if name=='read_prefill_only' else C['green'] if name=='read_periodic' else C['red'] if name=='read_all' else '#777777'
  error(ax,x,row,color,'s' if name=='read_all' else 'o');ax.annotate(label,(x,100*row['anls']),xytext=label_positions[name],textcoords='data',fontsize=7,
   arrowprops={'arrowstyle':'-','color':color,'lw':.65},bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':1.5})
 ax.axhline(100*allrow['anls'],ls='--',lw=1,color=C['red']);ax.set(xlabel='Residual + selection logical payload (% of all)',ylabel='ANLS (%)',xlim=(-5,118),ylim=(78,96),title='Fixed stored representation: refinement read trade-off')
 ax.yaxis.grid(alpha=.12);ax.set_axisbelow(True);save(fig,'confirm64_refinement_reads')
 captions['confirm64_refinement_reads']='All read policies own the same cap16 base+residual+selector metadata for each image and expose all base visual tokens. The x-axis sums declared tensor payloads and selection charges over free trajectories, not physical DRAM transactions. Output-length effects are separately controlled by fixed-token timing.95% image-bootstrap intervals.'
 fig,ax=plt.subplots(figsize=(7.4,3.1),layout='constrained');names=['oracle_lloyd_A00','oracle_lloyd_A10','oracle_lloyd_A01','full8']
 for i,name in enumerate(names):error(ax,i,table[name],C['blue'],'o')
 ax.set(xticks=range(4),xticklabels=['Quantized norm\n+ direction','Restore norm','Restore direction','Restore both'],ylabel='ANLS (%)',ylim=(0,100),title='Frozen64 mechanism replication: K factor intervention, V FP16')
 ax.yaxis.grid(alpha=.12);ax.set_axisbelow(True);save(fig,'confirm64_norm_direction')
 captions['confirm64_norm_direction']='All four arms own materialized FP16 K/V and are mechanism diagnostics. The quantized norm and direction are those of the actual Lloyd g128 reconstruction. Full8 provides the restore-both arm. Free dialog trajectories,286 answers per arm; no low-bit runtime claim from these oracle arms.'
 # Descriptive per-turn paired gaps control for different questions at each turn.
 import csv
 from analyze import paired
 scores=list(csv.DictReader((folder/'scores.csv').open(encoding='utf-8-sig')));groups={}
 for name in ['storage_primary','k2_lloyd_v1','calibration_warm','full8']:
  groups[name]={(r['image_id'],int(r['turn'])):float(r['score']) for r in scores if r['variant']==name}
 fig,ax=plt.subplots(figsize=(7.1,3.5),layout='constrained')
 for name,label,color in [('storage_primary','State-matched calibration',C['blue']),('k2_lloyd_v1','K2 / Lloyd V1',C['red']),('calibration_warm','Native-history calibration',C['teal'])]:
  means=[];low=[];high=[]
  for turn in range(1,7):
   keys={k:v for k,v in groups[name].items() if k[1]==turn};ref={k:v for k,v in groups['full8'].items() if k[1]==turn};p=paired(keys,ref,repeats=4000)
   means.append(100*p['delta']);low.append(100*p['ci_low']);high.append(100*p['ci_high'])
  xx=np.arange(1,7);ax.plot(xx,means,'o-',color=color,markersize=3,lw=1.3,label=label);ax.fill_between(xx,low,high,color=color,alpha=.10)
 ax.axhline(0,color='#666666',ls='--',lw=.8);ax.set(xlabel='Question index',ylabel='Paired ANLS gap to FP16 (pp)',xticks=range(1,7),title='Question-wise quality gaps in the frozen protocol');ax.legend(frameon=False,fontsize=8);save(fig,'confirm64_question_gaps')
 captions['confirm64_question_gaps']='Paired whole-image bootstrap bands, unadjusted95%. Questions differ across indices and some documents have fewer than6 questions. These curves describe protocol robustness and do not by themselves identify causal multi-turn error amplification.'
 (OUT/'captions.json').write_text(json.dumps(captions,indent=2));print(json.dumps({'figures':len(captions),'formats':['png','pdf','svg'],'output':str(OUT)}))
if __name__=='__main__':main()
