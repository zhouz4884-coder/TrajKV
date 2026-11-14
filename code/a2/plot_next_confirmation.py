"""Figures from the complete prospectively frozen Doc256/Text128 evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'figures/a3_confirmation'
BLUE='#0F4D92';TEAL='#42949E';RED='#B64342';GREEN='#8BCF8B';GREY='#777777';VIOLET='#9A4D8E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
 OUT.mkdir(exist_ok=True,parents=True)
 for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{suffix}',dpi=300)
 plt.close(fig)

def main():
 d=json.loads((ROOT/'results/next_confirm256_analysis/summary.json').read_text());assert d['sessions']==5888
 table={r['variant']:r for r in d['table']};captions={}
 fig,ax=plt.subplots(figsize=(7.1,3.8),layout='constrained')
 points=[('new_storage_primary','Frozen direction code',BLUE,'*'),('k2_lloyd_v1','K2 / Lloyd V1',TEAL,'o'),('previous_frozen_primary','Previous calibrated format',GREY,'v'),
  ('cold_multi_gain','Multi-token, state-matched',GREEN,'s'),('cold_multi_affine','Multi-token + center',VIOLET,'D'),('base2_8','Native2',RED,'o'),('native4_8','Native4',RED,'^')]
 for name,label,color,marker in points:
  r=table[name];y=100*r['anls'];ax.errorbar(r['effective_visual_bits'],y,yerr=[[y-100*r['ci_low']],[100*r['ci_high']-y]],fmt=marker,color=color,capsize=2,markersize=6,label=label)
 ax.axhline(100*table['full8']['anls'],color=GREY,ls='--',lw=1,label='FP16 cap8');ax.axvline(1.8,color=GREY,ls=':',lw=.8)
 ax.set(xlabel='Actual visual-cache bits / KV scalar',ylabel='ANLS (%)',xlim=(1.5,4.5),ylim=(81,97),title='Prospective Doc256: stored representation')
 ax.legend(frameon=False,fontsize=7,ncol=2,loc='lower right');ax.yaxis.grid(alpha=.12);save(fig,'doc256_storage_quality')
 captions['doc256_storage_quality']='256 source documents/999questions,cap8 inputs. Whole-source-document95%bootstrap intervals. All plotted low-bit methods use actual packed GPU readers. Full FP16 is a reference, not a mathematical accuracy ceiling. The frozen1.8bit/0.5pp storage quality target was not met; shared encoder constants are additionally charged in eligibility. KVarN materialized quality-port storage is excluded from the low-bit frontier.'
 target={r['variant']:r for r in d['read_targets']};fig,ax=plt.subplots(figsize=(7.1,3.6),layout='constrained')
 for name,label,color,marker in [('new_read_primary','Frozen primary',BLUE,'*'),('newread_cached','Cached selector',GREEN,'s'),('newread_freshrisk','Fresh selector',RED,'v'),
  ('newread_none','No refinement',GREY,'o'),('newread_periodic','Periodic',VIOLET,'^'),('newread_prefill_fresh','Prefill + fresh',TEAL,'D')]:
  r=target[name];x=100*(1-r['extra_read_saving']);y=100*r['delta']
  ax.errorbar(x,y,yerr=[[y-100*r['ci_low']],[100*r['ci_high']-y]],fmt=marker,color=color,capsize=2,markersize=6,label=label)
 ax.plot(100,0,'x',color='black',label='All refinement');ax.axhline(0,ls='--',lw=.8,color=GREY);ax.axhspan(-.5,.5,color='#CFCECE',alpha=.22)
 ax.set(xlabel='Extra logical read payload (% of all)',ylabel='Paired ANLS gap to all (pp)',xlim=(-5,108),ylim=(-4,2),title='Prospective Doc256: same stored cache,cap16')
 ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper right');ax.yaxis.grid(alpha=.12);save(fig,'doc256_read_quality')
 captions['doc256_read_quality']='All read policies own the same4.0679-bit representation per image. Frozen primary uses60.62%fewer extra logical read bytes and has a−0.312pp point gap to all. The point target passes; the95%paired interval[−0.818,+0.165]pp does not establish0.5pp noninferiority. CIs unadjusted; frozen primary superiority tests are Holm corrected separately. Logical tensor payload is not HBM; free output-length differences are controlled separately in timing.'
 dev=json.loads((ROOT/'results/cached_reads_factorial64_analysis/summary.json').read_text());dv={r['variant']:r['anls'] for r in dev['table']}
 values=[[[100*(dv[n]-dv['cachedread_all']) for n in ['cachedread_risk50','cachedread_cached50']],
   [100*(dv[n]-dv['cachedread_all']) for n in ['cachedread_prefill_freshrisk50','cachedread_prefill_cached50']]],
  [[100*(table[n]['anls']-table['newread_all']['anls']) for n in ['newread_freshrisk','newread_cached']],
   [100*(table[n]['anls']-table['newread_all']['anls']) for n in ['newread_prefill_fresh','new_read_primary']]]]
 fig,axes=plt.subplots(1,2,figsize=(6.4,2.9),layout='constrained')
 for ax,arr,title in zip(axes,values,['Open dev64','Prospective confirm256']):
  im=ax.imshow(arr,vmin=-1.5,vmax=1.5,cmap='RdBu',aspect='auto')
  for i,row in enumerate(arr):
   for j,value in enumerate(row):ax.text(j,i,f'{value:+.2f}',ha='center',va='center',fontsize=11,color='white' if abs(value)>1 else '#222222')
  ax.set(xticks=[0,1],xticklabels=['Fresh','Cached'],yticks=[0,1],yticklabels=['Budgeted prefill','Protect prefill'],title=title)
 fig.colorbar(im,ax=axes,shrink=.8,label='ANLS gap to all (pp)');save(fig,'prefill_effect_development_vs_confirmation')
 captions['prefill_effect_development_vs_confirmation']='2x2phase/reuse ablations at the0.5byte-budget setting,16full-residual-equivalent global cap. Values are within-dataset gaps to all, not cross-dataset absolute quality. The added-prefill benefit seen in development did not replicate; frozen primary labels are retained. Selection fees and later budget exhaustion also affect realized precision coverage.'
 external=json.loads((ROOT/'results/next_textbalanced128_analysis/summary.json').read_text());et={r['variant']:r for r in external['table']};assert external['records']==1664
 fig,ax=plt.subplots(figsize=(6.9,3.4),layout='constrained')
 names=['full8','native4_8','base2_8','new_storage_primary','k2_lloyd_v1','cold_multi_gain'];labels=['FP16','Native4','Native2','Direction','Lloyd V1','Calibrated']
 for offset,key,label,color in [(-.15,'numeric_accuracy','Digit-containing majority answers',BLUE),(.15,'text_accuracy','Nonbinary text majority answers',TEAL)]:
  ax.plot([i+offset for i in range(len(names))],[100*et[n][key] for n in names],'o',color=color,label=label)
 ax.set(xticks=range(len(names)),xticklabels=labels,ylabel='TextVQA soft accuracy (%)',ylim=(60,95),title='External 128: predefined 64 + 64 answer strata')
 ax.legend(frameon=False,fontsize=7,loc='lower left');ax.yaxis.grid(alpha=.12);save(fig,'external128_answer_strata')
 captions['external128_answer_strata']='Fresh current-cycle128images selected before outcomes,64digit-containing-majority answers and64nonbinary-text-majority answers. Official MMF ten-annotator soft accuracy, not ANLS; strata are not an overall official-distribution estimate. These are joint K/V interventions, so stratum gaps alone cannot attribute sensitivity specifically to V. The DocVQA direction-code advantage did not become a universal external advantage.'
 timing=json.loads((ROOT/'results/next_confirm_timing/summary.json').read_text());assert timing['status']=='PASS' and timing['measured_trials']==288
 tt={r['variant']:r for r in timing['table']}
 groups=[('Stored representation, cap8',[('full8','FP16'),('native4_8','Native4'),('base2_8','Native2'),('k2_lloyd_v1','K2 / Lloyd V1'),('warm_multi_gain','Calibrated'),('new_storage_primary','Direction code')]),
  ('Fixed storage, cap16',[('newread_all','All refinement'),('newread_none','No refinement'),('newread_periodic','Periodic'),('newread_cached','Cached selector'),('new_read_primary','Frozen primary'),('newread_prefill_fresh','Prefill + fresh'),('newread_freshrisk','Fresh selector')])]
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),layout='constrained')
 for ax,(title,group) in zip(axes,groups):
  for i,(name,label) in enumerate(group):
   row=tt[name];med=row['session_ms_median']/1000;low=row['session_ms_p25']/1000;high=row['session_ms_p75']/1000
   color=BLUE if name in ('new_storage_primary','new_read_primary') else RED if name=='newread_freshrisk' else TEAL
   ax.errorbar(med,i,xerr=[[med-low],[high-med]],fmt='o',color=color,capsize=2,markersize=4)
  ax.set(yticks=range(len(group)),yticklabels=[x[1] for x in group],xlabel='Session time (s), median and IQR',title=title)
  ax.invert_yaxis();ax.xaxis.grid(alpha=.12)
 save(fig,'fixed_tokens_timing')
 captions['fixed_tokens_timing']='Six development images, three measured repeats per method, eight identical forced output IDs per question. Total 288 measured trials plus 96 warmups across 16 methods; 13 shown here. Input preparation, causal-prefix extraction, encoding and question replay are included; model loading is excluded. Both GPUs were otherwise quiet at each trial boundary. Bars are pooled median/IQR, not confidence intervals. Cap8 and cap16 use different image inputs. The frozen read primary had a 3.082s median versus 3.036s for all; lower logical reads did not establish end-to-end acceleration. These forced trials are never scored as task accuracy.'
 (OUT/'captions.json').write_text(json.dumps(captions,indent=2));print(json.dumps({'figures':len(captions),'output':str(OUT)}))
if __name__=='__main__':main()
