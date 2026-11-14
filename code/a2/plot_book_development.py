"""Completed development-only figures; no partial confirmation scores read."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'figures/a4_codebooks_development'
BLUE='#0F4D92';TEAL='#42949E';RED='#B64342';GREY='#777777';GREEN='#8BCF8B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
 OUT.mkdir(exist_ok=True,parents=True)
 for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{suffix}',dpi=300)
 plt.close(fig)

def main():
 d=json.loads((ROOT/'results/book_factorial64_analysis/factorial_summary.json').read_text());assert d['sessions']==960 and d['raw_gpu_sessions']==1152
 table={r['variant']:r for r in d['table']};base=table['fwht_global']['anls'];captions={}
 fig,axes=plt.subplots(1,2,figsize=(6.5,2.8),layout='constrained')
 for ax,condition,title in zip(axes,['cold','warm'],['Compressed question','Native question history']):
  name='decoder_'+condition+'_joint';names=[['fwht_global',name+'_vonly'],[name+'_konly',name]]
  gaps=np.array([[100*(table[n]['anls']-base) for n in row] for row in names]);im=ax.imshow(gaps,vmin=-2.7,vmax=2.7,cmap='RdBu',aspect='auto')
  for i,row in enumerate(names):
   for j,n in enumerate(row):ax.text(j,i,f"{100*table[n]['anls']:.2f}\n({gaps[i,j]:+.2f})",ha='center',va='center',fontsize=10,color='white' if abs(gaps[i,j])>2 else '#222222')
  ax.set(xticks=[0,1],xticklabels=['Base V','Calibrated V'],yticks=[0,1],yticklabels=['Global K book','Learned K book'],title=title)
 fig.colorbar(im,ax=axes,shrink=.8,label='ANLS change to untrained base (pp)');save(fig,'kv_joint_fit_factorial')
 captions['kv_joint_fit_factorial']='Open development64 source images/255questions, 15 unique configurations from1152 actual GPU free sessions including192 exact repeated controls. Each2x2 toggles K/V parameter blocks from the same jointly fitted artifact, without retraining. Labels give ANLS and change to the global-book base. V-only uses a compact global decoder, so its actual stored price is1.6946bits including encoder constants versus1.7302for joint; numerical base interpretation was controlled. These are development findings, not confirmation. Point interactions do not establish additive causal loss shares.'
 p=json.loads((ROOT/'results/book_continuous64_analysis/summary.json').read_text());assert p['proxy_records']==704;pt={r['variant']:r for r in p['table']}
 names=['native4_8','base2_8','fwht_global','decoder_cold_book_only','decoder_warm_book_only','decoder_cold_joint','decoder_warm_joint']
 labels=['Native4','Native2','Untrained','K, compressed Q','K, native history','K+V, compressed Q','K+V, native history']
 fig,ax=plt.subplots(figsize=(6.5,3.0),layout='constrained')
 for i,n in enumerate(names):ax.plot(pt[n]['mean_kl_fp16_to_method'],i,'o',color=BLUE if n.endswith('joint') else TEAL if 'book_only' in n else GREY,markersize=5)
 ax.set(yticks=range(len(names)),yticklabels=labels,xlabel='Mean FP16-to-method vocabulary KL',xlim=(-.004,.11),title='First-question held-out diagnostics (64 images)');ax.invert_yaxis();ax.xaxis.grid(alpha=.12);save(fig,'heldout_vocabulary_kl')
 captions['heldout_vocabulary_kl']='Canonical first-reference teacher contexts on64 development images,704 proxy records,397positions includingEOS. KL averages content positions within each image then images; EOS excluded. Actual GPU full-model forwards, not free answers. K-only calibration did not materially reduce this held-out KL; joint fits reduced it. No error bars or training-to-test KL ratio is implied: calibration uses self-generated teacher targets and different contexts.'
 g=json.loads((ROOT/'results/book_geometry16_analysis/summary.json').read_text());assert g['status']=='PASS';gt={r['variant']:r for r in g['table']}
 layers=list(csv.DictReader((ROOT/'results/book_geometry16_analysis/layer_metrics.csv').open(encoding='utf-8-sig')))
 fig,(left,right)=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained',gridspec_kw={'width_ratios':[.9,1.4]})
 for i,(n,label,color) in enumerate([('fwht_global','Global / V-only',GREY),('decoder_warm_book_only','Learned K only',TEAL),('decoder_warm_joint','Joint fit',BLUE)]):
  row=gt[n];v=row['centered_key_cosine'];left.errorbar(v,i,xerr=[[v-row['centered_key_cosine_ci_low']],[row['centered_key_cosine_ci_high']-v]],fmt='o',capsize=2,color=color)
 left.set(yticks=[0,1,2],yticklabels=['Global / V-only','Learned K only','Joint fit'],xlabel='Centered-key cosine',title='Key reconstruction');left.invert_yaxis();left.xaxis.grid(alpha=.12)
 for n,label,color in [('fwht_global','Untrained base',GREY),('decoder_warm_joint_vonly','V-only applied',TEAL),('decoder_warm_joint','Joint applied',BLUE)]:
  rows=sorted([r for r in layers if r['variant']==n],key=lambda x:int(x['layer']));right.plot([int(r['layer'])+1 for r in rows],[1-float(r['query_cosine_to_full']) for r in rows],label=label,color=color,lw=1.2)
 right.set(xlabel='Transformer layer',ylabel='1 - cosine to Full query',title='Actual last-prefill queries');right.yaxis.grid(alpha=.12);right.legend(frameon=False,fontsize=7)
 save(fig,'key_geometry_and_endogenous_queries')
 captions['key_geometry_and_endogenous_queries']='First16 fixed development inputs, actual packed first-question last-prefill queries,7 methods plus Full. Centered-key cosine uses image means over28layers; left error bars are source-image bootstrap95%intervals. Right shows layer means over16 images. V-only and untrained have exactly identical reconstructed K and fixed-Full-query key-score diagnostics in all448 image/layer pairs, yet their endogenous queries differ because V changes earlier hidden states. Extra FP16 source keys exist only in this diagnostic; no memory, task-accuracy or speed claim. Raw score RMSE is uncentered and cannot establish state dominance or softmax disruption.'
 (OUT/'captions.json').write_text(json.dumps(captions,indent=2));print(json.dumps({'figures':len(captions),'output':str(OUT)}))

if __name__=='__main__':main()
