"""Complete external replication and paired state-condition effects."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
  'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
 doc=json.loads((ROOT/'results/frozen_confirm64_analysis/summary.json').read_text());ext=json.loads((ROOT/'results/frozen_external128_analysis/summary.json').read_text())
 assert doc['sessions']==2112 and ext['records']==1664
 table={r['variant']:r for r in ext['table']};fig,axes=plt.subplots(1,2,figsize=(7.3,3.1),layout='constrained')
 names=['native_k2v1_8','k2_lloyd_v1','calibration_warm','storage_primary'];labels=['Native K2/V1','Lloyd V1','Native-history\ncalibration','State-matched\ncalibration']
 colors=['#CFCECE','#B64342','#42949E','#0F4D92']
 ax=axes[0]
 for i,(name,color) in enumerate(zip(names,colors)):
  row=table[name];y=100*row['vqa_accuracy'];ax.errorbar(i,y,yerr=[[y-100*row['ci_low']],[100*row['ci_high']-y]],fmt='o',color=color,capsize=2.5)
 ax.axhline(100*table['full8']['vqa_accuracy'],color='#555555',ls='--',lw=1,label='FP16')
 ax.set(xticks=range(4),xticklabels=labels,ylabel='TextVQA soft accuracy (%)',ylim=(65,100),title='External128: same 1.752-bit cache')
 ax.tick_params(axis='x',labelsize=7);ax.legend(frameon=False,fontsize=7);ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True)
 ax=axes[1]
 d=next(r for r in doc['primary_comparisons'] if r['left']=='storage_primary' and r['right']=='calibration_warm') if 'primary_comparisons' in doc else None
 if d is None:
  import csv
  d=next(r for r in csv.DictReader((ROOT/'results/frozen_confirm64_analysis/primary_comparisons.csv').open(encoding='utf-8-sig')) if r['left']=='storage_primary' and r['right']=='calibration_warm')
 e=next(r for r in ext['comparisons'] if r['candidate']=='storage_primary' and r['comparator']=='calibration_warm')
 for i,(row,color) in enumerate([(d,'#0F4D92'),(e,'#42949E')]):
  y=100*float(row['delta']);lo=100*float(row['ci_low']);hi=100*float(row['ci_high'])
  ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color,capsize=3)
  ax.annotate(f'+{y:.2f}',(i,y),xytext=(8,0),textcoords='offset points',va='center',fontsize=8)
 ax.axhline(0,color='#777777',lw=.8,ls='--');ax.set(xticks=[0,1],xticklabels=['DocVQA64\nANLS','TextVQA128\nsoft accuracy'],ylabel='State-matched minus native-history (pp)',
  xlim=(-.4,1.6),ylim=(-1,14),title='Calibration-condition replication');ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True)
 out=ROOT/'figures/a2_confirmation';out.mkdir(exist_ok=True,parents=True)
 for ext in ('png','pdf','svg'):fig.savefig(out/f'external_state_replication.{ext}',dpi=300)
 plt.close(fig)
 (out/'external_state_replication_caption.txt').write_text('Left:128 images,13 fixed methods,official ten-annotator TextVQA soft accuracy. Shown methods use the same1.752-bit packed cache format. Right:paired whole-image95% bootstrap intervals; DocVQA and TextVQA use distinct task metrics, displayed as percentage-point differences within each dataset. External comparisons are secondary replications; DocVQA condition comparison belongs to the preregistered six-test primary family.',encoding='utf-8')
 print(json.dumps({'figure':'external_state_replication','formats':['png','pdf','svg']}))
if __name__=='__main__':main()
