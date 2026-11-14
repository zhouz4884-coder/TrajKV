"""Standalone scientific figures for completed open-development experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'figures/a3_development'
C={'blue':'#0F4D92','green':'#8BCF8B','red':'#B64342','teal':'#42949E','violet':'#9A4D8E','grey':'#777777'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def data(path):return json.loads((ROOT/path).read_text())
def save(fig,name):
 OUT.mkdir(exist_ok=True,parents=True)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 plt.close(fig)

def main():
 dev=data('results/trajectory_development64_analysis/summary.json');reads=data('results/cached_reads_factorial64_analysis/summary.json');density=data('results/density_development64_analysis/summary.json');audit=data('results/development_mechanism_audit/summary.json')
 assert dev['sessions']==896 and reads['sessions']==640
 table={r['variant']:r for r in dev['table']};captions={}
 fig,ax=plt.subplots(figsize=(6.3,3.4),layout='constrained')
 for prefix,label,color,offset in [('cold','Compressed-question states',C['blue'],-.06),('warm','Native-question history',C['teal'],.06)]:
  y=[100*table[prefix+'_'+name]['anls'] for name in ['first_gain','multi_gain','multi_affine']]
  ax.plot([x+offset for x in range(3)],y,'o-',color=color,label=label,lw=1.4)
  for i,value in enumerate(y):ax.annotate(f'{value:.2f}',(i+offset,value),xytext=(0,7 if prefix=='cold' else -13),textcoords='offset points',ha='center',fontsize=7)
 ax.axhline(100*table['full8']['anls'],color=C['grey'],ls='--',lw=1,label='FP16')
 ax.set(xticks=range(3),xticklabels=['First-token\n224 parameters','Up to8 tokens\n224 parameters','Up to8 + center\n448 parameters'],ylabel='ANLS (%)',ylim=(79,95),title='Open dev64: state condition x calibration target')
 ax.legend(frameon=False,fontsize=7,loc='upper right');ax.yaxis.grid(alpha=.15);save(fig,'calibration_condition_development')
 captions['calibration_condition_development']='64 source-disjoint development documents/255questions. Fits use32 different source documents/83unlabelled questions,400updates,identity K2 initialization. Lines are point estimates; paired uncertainty is in the accompanying comparisons.csv. Center arms change parameter count and are separate ablations. Not prospective confirmation.'
 rt={r['variant']:r for r in reads['table']};targets={r['variant']:r for r in reads['read_targets']}
 fig,axes=plt.subplots(1,2,figsize=(8.0,3.6),layout='constrained');ax=axes[0]
 selected=[('cachedread_none','None',C['grey'],'o'),('cachedread_risk50','Fresh risk',C['red'],'v'),('cachedread_cached50','Cached risk',C['green'],'s'),
  ('cachedread_prefill_freshrisk50','Prefill + fresh',C['teal'],'D'),('cachedread_prefill_cached50','Prefill + cached',C['blue'],'*'),('cachedread_periodic25','Periodic',C['violet'],'^')]
 for name,label,color,marker in selected:
  r=targets[name];x=100*(1-r['extra_read_saving']);y=100*r['delta']
  ax.errorbar(x,y,yerr=[[y-100*r['ci_low']],[100*r['ci_high']-y]],fmt=marker,color=color,capsize=2,markersize=6,label=label)
 ax.plot(100,0,'x',color='black',label='All refinement');ax.axhline(0,color=C['grey'],ls='--',lw=.8);ax.axhspan(-.5,.5,color='#CFCECE',alpha=.2)
 ax.set(xlabel='Extra logical payload (% of all)',ylabel='Paired ANLS gap to all (pp)',title='Same stored cache, open dev64',xlim=(-5,108),ylim=(-4.2,4.5));ax.legend(frameon=False,fontsize=6.6,ncol=2,loc='upper right');ax.yaxis.grid(alpha=.12)
 cov={(r['variant'],r['phase']):100*r['actual_selected_page_fraction'] for r in audit['coverage']};names=['cachedread_risk50','cachedread_cached50','cachedread_prefill_freshrisk50','cachedread_prefill_cached50']
 values=[[cov[n,p] for p in ['prefill','decode']] for n in names];ax=axes[1];im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
 for i,row in enumerate(values):
  for j,value in enumerate(row):ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>60 else '#222222',fontsize=8)
 ax.set(xticks=[0,1],xticklabels=['Question prefill','Decode'],yticks=range(4),yticklabels=['Fresh','Cached','Prefill + fresh','Prefill + cached'],title='Actual refined head-pages')
 ax.tick_params(axis='y',length=0,labelsize=7);fig.colorbar(im,ax=ax,shrink=.8,label='Coverage (%)');save(fig,'read_budget_and_phase_development')
 captions['read_budget_and_phase_development']='All four0.5budget methods hold identical4.0679-bit storage and16full-residual-equivalent global budgets.0.5 denotes a byte cap including selection costs, not50% page coverage. Overhead and budget exhaustion change actual coverage. Left intervals:95% whole-document paired bootstrap, exploratory. Logical payload is not physical DRAM; free trajectories have different output lengths. First-question prefill logits match in all64 protected-fresh/protected-cached pairs.'
 fig,ax=plt.subplots(figsize=(7.3,3.8),layout='constrained')
 points=[('base2_8','Base2 cap8',C['red'],'o',dev),('native4_8','Native4 cap8',C['grey'],'o',dev),('direction_vq2_v1','Direction cap8',C['blue'],'o',dev),('warm_multi_gain','Calibrated cap8',C['teal'],'o',dev),
  ('base2_cap16','Base2 cap16',C['red'],'s',density),('native4_cap16','Native4 cap16',C['grey'],'s',density),('warm_multi_gain_cap16','Calibrated cap16',C['teal'],'s',density),('direction_vq2_v1_cap16','Direction cap16',C['blue'],'s',density)]
 for name,label,color,marker,source in points:
  r=next(x for x in source['table'] if x['variant']==name);x=r['visual_bytes_total']/64/(1024**2);ax.plot(x,100*r['anls'],marker,color=color,label=label,markersize=5)
 ax.axhline(100*table['full8']['anls'],ls='--',lw=.8,color=C['grey'],label='FP16 cap8');ax.set(xlabel='Mean actual visual-cache storage (MiB)',ylabel='ANLS (%)',ylim=(80,95),title='Resolution and precision: actual-byte comparison')
 ax.legend(frameon=False,fontsize=7,ncol=2,loc='lower right');ax.yaxis.grid(alpha=.15);save(fig,'density_actual_bytes_development')
 captions['density_actual_bytes_development']='Same64 source documents/255questions, different image-input caps shown separately. Actual backing includes required metadata/constants/positions; model weights and common FP16 text capacity are separate. More tokens can lower bits/scalar while increasing total storage. Do not infer a universal resolution benefit or optimal rate from these exploratory points.'
 (OUT/'captions.json').write_text(json.dumps(captions,indent=2));print(json.dumps({'figures':len(captions),'output':str(OUT)}))
if __name__=='__main__':main()
