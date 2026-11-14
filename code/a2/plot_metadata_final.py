"""Final bounded-study figures from complete checked GPU evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'figures/a5_metadata_confirmation'
BLUE='#0F4D92';TEAL='#42949E';RED='#B64342';GREY='#777777'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
 OUT.mkdir(exist_ok=True,parents=True)
 for extension in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{extension}',dpi=300)
 plt.close(fig)

def main():
 d=json.loads((ROOT/'results/metadata_confirm_doc128_analysis/summary.json').read_text());assert d['sessions']==1024 and d['audit']['status']=='PASS'
 table={row['variant']:row for row in d['table']};primary=d['primary_candidate'];cheap=d['secondary_low_metadata'];captions={}
 fig,(left,right)=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
 points=[('metadata_identity_all','Whole-image V, uncalibrated',GREY,'o'),(cheap,'Whole-image V, calibrated',TEAL,'s'),('fwht_global','g128, uncalibrated',GREY,'^'),(primary,'g128, calibrated',BLUE,'*'),('base2_8','Native2',RED,'o')]
 for name,label,color,marker in points:
  r=table[name];mean=100*r['anls'];left.errorbar(r['budget_bits_including_encoder_constants'],mean,yerr=[[mean-100*r['ci_low']],[100*r['ci_high']-mean]],fmt=marker,color=color,capsize=2,markersize=5,label=label)
 left.axhline(100*table['native4_8']['anls'],color=RED,ls='--',lw=.8,label='Native4 quality (4.25 bits)')
 left.set(xlabel='Visual bits / KV scalar',ylabel='ANLS (%)',xlim=(1.5,2.35),ylim=(78,97),title='Fresh 128 source documents / 485 QA');left.legend(frameon=False,fontsize=6.2,loc='lower right');left.yaxis.grid(alpha=.12)
 tests={(r['left'],r['right']):r for r in d['primary_comparisons']}
 for i,(name,base,label,color) in enumerate([(primary,'fwht_global','g128 calibration',BLUE),(cheap,'metadata_identity_all','Whole-image calibration',TEAL)]):
  r=tests[name,base];mean=100*r['delta'];right.errorbar(mean,i,xerr=[[mean-100*r['ci_low']],[100*r['ci_high']-mean]],fmt='o',color=color,capsize=2)
  right.text(mean,i+.17,f"Holm p={r['holm_family']:.4f}",fontsize=7,ha='center')
 right.axvline(0,color=GREY,lw=.8,ls='--');right.set(yticks=[0,1],yticklabels=['g128','Whole-image'],xlabel='Paired ANLS change (pp)',title='Same encoded format, calibrated');right.invert_yaxis();right.set_ylim(1.5,-.5);right.xaxis.grid(alpha=.12)
 save(fig,'grouping_and_calibration_fresh128')
 captions['grouping_and_calibration_fresh128']='128 new source documents,485questions,8 frozen methods,1024 complete actual packed GPU free sessions. Whole-source-document bootstrap95%intervals. The calibrated whole-image grouping gained3.574pp over its same-cache-format uncalibrated control, CI[1.514,5.714]pp, six-test Holm p=.00372. The g128 contrast gained2.123pp, Holm p=.221. Calibration adds1344B encoder-only constants, folded into existingF16 metadata; they are included in the x-axis qualification cost. Native4 quality line lies outside the plotted storage range. Fixed Native4/0.5pp storage target remains unmet.'
 transfer=json.loads((ROOT/'results/metadata_text256_transfer_analysis/summary.json').read_text());tt={r['variant']:r for r in transfer['table']}
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained')
 for ax,source,metric,title in [(axes[0],d['resource_comparisons'],'delta','Fresh Doc128 (485 QA)'),(axes[1],transfer['comparisons'],'delta','Reused Text256 transfer')]:
  for name,label,color,marker in [(primary,'g128 calibrated',BLUE,'o'),(cheap,'Whole-image calibrated',TEAL,'s')]:
   r=next(row for row in source if row['candidate']==name and row['reference']=='base2_8');x=table[name]['budget_bits_including_encoder_constants'] if ax is axes[0] else tt[name]['budget_bits_including_encoder_constants'];mean=100*r[metric]
   ax.errorbar(x,mean,yerr=[[mean-100*r['ci_low']],[100*r['ci_high']-mean]],fmt=marker,color=color,capsize=2,markersize=5,label=label)
  ax.plot(2.252232142857,0,'x',color=RED,label='Native2 reference');ax.axhline(0,color=GREY,lw=.8,ls='--');ax.set(xlim=(1.5,2.35),ylim=(-6,5),xlabel='Visual bits / KV scalar',title=title);ax.yaxis.grid(alpha=.12);ax.legend(frameon=False,fontsize=7,loc='lower right')
 axes[0].set_ylabel('Paired quality gap to Native2 (pp)');save(fig,'metadata_resource_quality_two_cohorts')
 captions['metadata_resource_quality_two_cohorts']='Secondary comparisons to the fixed Native2 configuration. Left uses ANLS on independent128 source documents/485questions; right uses official10-annotator soft accuracy on a previously evaluated balanced TextVQA256 cohort, with candidates chosen only on Doc development. The g128 method saves24.75%visual storage and has−0.888pp Doc/+0.820pp Text point gaps; whole-image saves29.90%with−2.158pp Doc/+0.391pp Text.95%paired intervals shown; crossing zero does not establish equivalence. The primary strong-reference target is Native4 and is reported separately.'
 timing=json.loads((ROOT/'results/metadata_confirm_timing/summary.json').read_text());assert timing['status']=='PASS' and timing['measured_trials']==36;lt={r['variant']:r for r in timing['table']}
 fig,ax=plt.subplots(figsize=(6.3,2.8),layout='constrained')
 names=[('full8','FP16'),('native4_8','Native4'),('base2_8','Native2'),('fwht_global','Uncalibrated global'),(primary,'Calibrated g128'),(cheap,'Calibrated whole-image')]
 for i,(name,label) in enumerate(names):
  row=lt[name];mean=row['session_ms_median']/1000;lo=row['session_ms_p25']/1000;hi=row['session_ms_p75']/1000
  ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=BLUE if name==primary else TEAL if name==cheap else GREY,capsize=2)
 ax.set(yticks=range(len(names)),yticklabels=[label for _,label in names],xlabel='Session seconds, pooled median and IQR',title='Fixed output IDs: 2 development images, 3 repeats');ax.invert_yaxis();ax.xaxis.grid(alpha=.12);save(fig,'metadata_fixed_output_timing')
 captions['metadata_fixed_output_timing']='Small descriptive full-pipeline timing,2fixed development images,6methods,3measured repeats=36measurements plus12warmups. Every question consumes identical8forced output IDs; no accuracy is scored. Both GPUs otherwise quiet at trial boundaries. Model loading excluded; image preparation, native causal-prefix extraction, encoding, question replay and decode included. Calibratedg128 median2.641s versus Native2 2.426s, about8.84%slower; metadata folding is close to uncalibratedglobal2.632s. IQR is not a confidence interval; no throughput, physical HBM or general speedup claim.'
 (OUT/'captions.json').write_text(json.dumps(captions,indent=2));print(json.dumps({'figures':len(captions),'directory':str(OUT)}))

if __name__=='__main__':main()
