"""Evidence-only figures: matched100 factorial mechanism and continuous loss.

Python/matplotlib; exports vector PDF/SVG plus 300dpi PNG. No imputed points.
"""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/a2';OUT.mkdir(exist_ok=True,parents=True)
COLORS={'blue':'#0F4D92','green':'#8BCF8B','red':'#B64342','teal':'#42949E','violet':'#9A4D8E','grey':'#CFCECE'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'savefig.dpi':300,'axes.linewidth':.7,'xtick.major.width':.6,'ytick.major.width':.6})
def save(fig,name):
 for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 plt.close(fig)
def main():
 table={r['variant']:r for r in json.loads((ROOT/'results/radial_oracle100_analysis/summary.json').read_text())['table']}
 energies=json.loads((ROOT/'results/radial_oracle100_analysis/mechanism.json').read_text())['energies']
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.55),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
 ax=axes[0]
 for enc,color,offset,marker in [('mad',COLORS['blue'],-.035,'o'),('lloyd',COLORS['violet'],.035,'s')]:
  rows=[table[enc+'_A00'],table[enc+'_A10'],table[enc+'_A01'],table['A11_fp16']]
  mean=100*np.array([r['anls'] for r in rows]);low=100*np.array([r['ci_low'] for r in rows]);high=100*np.array([r['ci_high'] for r in rows])
  ax.errorbar(np.arange(4)+offset,mean,yerr=[mean-low,high-mean],fmt=marker+'-',color=color,linewidth=1.5,markersize=4,capsize=2.5,label=enc.upper() if enc=='mad' else 'Lloyd')
 ax.set(xticks=np.arange(4),xticklabels=['Quantized\nboth','Restore\nnorm','Restore\ndirection','Restore\nboth'],ylabel='ANLS (%)',ylim=(35,100),title='a  Direction recovery gives the larger gain')
 ax.legend(loc='lower right',frameon=False);ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True)
 ax=axes[1];xx=np.arange(len(energies));rad=100*np.array([r['radial_energy_share'] for r in energies]);ang=100-rad
 ax.bar(xx,ang,color=COLORS['red'],width=.55,label='Angular term');ax.bar(xx,rad,bottom=ang,color=COLORS['teal'],width=.55,label='Radial term')
 for i,(a,b) in enumerate(zip(ang,rad)):ax.text(i,a/2,f'{a:.1f}%',ha='center',va='center',color='white',weight='bold');ax.text(i,102,f'{b:.1f}% radial',ha='center',fontsize=8)
 ax.set(xticks=xx,xticklabels=['MAD','Lloyd'],ylim=(0,113),ylabel='Share of key squared error (%)',title='b  Measured error decomposition')
 ax.legend(frameon=False,loc='lower center',fontsize=8)
 save(fig,'mechanism100_norm_direction')
 free={r['variant']:r for r in json.loads((ROOT/'results/mad100_analysis/summary.json').read_text())['table']}
 continuous={r['variant']:r for r in json.loads((ROOT/'results/mad100_continuous_analysis/summary.json').read_text())['table']}
 names=['rtn_k1v1','rtn_k1_vmadc','mad_k1_vmadc','mad_k1_vmadc_headgain','native_k2v1','base2_reference']
 labels=['RTN K1/V1','RTN K1 / MAD V1','MAD K1/V1','MAD + head LS','Native K2/V1','Native K2/V2']
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.55),layout='constrained')
 xx=np.arange(len(names));colors=[COLORS['grey'],COLORS['grey'],COLORS['blue'],COLORS['teal'],COLORS['violet'],COLORS['red']]
 axes[0].barh(xx,[100*free[n]['anls'] for n in names],color=colors,height=.62)
 axes[0].set(yticks=xx,yticklabels=labels,xlim=(0,100),xlabel='Free-generation ANLS (%)',title='a  Joint packed visual KV, 100 images');axes[0].invert_yaxis()
 axes[1].barh(xx,[continuous[n]['mean_kl_fp16_to_method'] for n in names],color=colors,height=.62)
 axes[1].set(yticks=xx,yticklabels=[],xlabel='Mean FP16-to-method logit KL (nats)',title='b  Continuous degradation is also present');axes[1].invert_yaxis()
 for ax in axes:ax.xaxis.grid(alpha=.15);ax.set_axisbelow(True)
 save(fig,'mechanism100_joint_quality_continuous')
 captions={'mechanism100_norm_direction':'100 distinct development images, one question each. V remains original FP16. A00 uses actual quantized norms and directions; A10 restores only norms, A01 only directions, A11 both. All four are FP16-materialized diagnostic arms, not low-bit deployment candidates. Error bars are 95% image bootstrap intervals. Energy shares are computed from the exact radial/angular squared-error identity; they are not task-causal percentages.',
  'mechanism100_joint_quality_continuous':'Same 100 development images and official short-answer prompt. Left: free end-to-end GPU generation with actual packed joint KV. Right: canonical-reference teacher-forced logit KL, separate GPU forwards at identical gold contexts; never treated as free-answer accuracy. Metadata and gain costs differ and are reported in the resource table.'}
 (OUT/'mechanism_captions.json').write_text(json.dumps(captions,indent=2))
 print(json.dumps({'figures':2,'formats':['png','pdf','svg'],'output':str(OUT)}))
if __name__=='__main__':main()
