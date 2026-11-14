"""Prespecified same100 codec contrasts and observed packed-runtime frontier."""
import csv,json
from pathlib import Path
from analyze import paired,write_csv
ROOT=Path(__file__).resolve().parents[2]
def main():
 folder=ROOT/'results/final100_analysis';summary=json.loads((folder/'summary.json').read_text());table=summary['table'];by={}
 for r in csv.DictReader((folder/'scores.csv').open(encoding='utf-8-sig')):by.setdefault(r['variant'],{})[(r['image_id'],1)]=float(r['anls'])
 pairs=[('k2_vlloyd_channel','native_k2v1'),('k2_vlloyd_channel','k2_vrtn_channel'),('k2_vlloyd_channel','k2_vlloyd_token'),
        ('k2_vmad_global','native_k2v1'),('k2_vlloyd_channel','base2_reference'),('k2_vlloyd_channel','fp16_reference100'),
        ('lloyd_joint_centerls','lloyd_joint_none'),('mad_k1_vmadc_headgain','mad_k1_vmadc'),
        ('mad_origin_norm','mad_k1_vmadc'),('mad_global_head','mad_k1_vmadc_headgain')]
 contrasts=[{'left':left,'right':right,**paired(by[left],by[right])} for left,right in pairs]
 actual=[r for r in table if r['role'] in ('actual packed runtime','FP16 runtime')];frontier=[]
 for r in sorted(actual,key=lambda z:(z['effective_bits'],-z['anls'])):
  if not any(t['effective_bits']<=r['effective_bits'] and t['anls']>=r['anls'] and (t['effective_bits']<r['effective_bits'] or t['anls']>r['anls']) for t in actual):frontier.append(r)
 out={'contrasts':contrasts,'empirical_frontier':frontier,'scope':'old-image development100; finite observed points, not theoretical optimum; materialized KVarN quality port excluded from resource frontier'}
 (folder/'contrasts_frontier.json').write_text(json.dumps(out,indent=2));write_csv(folder/'paired_contrasts.csv',contrasts);write_csv(folder/'empirical_frontier.csv',frontier)
 print(json.dumps(out))
if __name__=='__main__':main()
