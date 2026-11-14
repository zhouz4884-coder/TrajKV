"""Freeze one final fresh-source check after bounded round-five exploration."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from remote import connect,upload,REMOTE_ROOT as R,PYTHON as P


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    marker=ROOT/'audit/a2/round5_confirmation_v1.json'
    if marker.exists():raise FileExistsError(marker)
    folder=ROOT/'configs/a2/round5_v1'
    tables={}
    summaries=['development64_strong_analysis','mixed_dev64_analysis','policy_dev64_analysis','gain_dev64_analysis']
    for name in summaries:
        data=load(ROOT/'results/round5_v1'/name/'summary.json')
        if data['images']!=64 or data['questions_per_variant']!=255:raise ValueError('Incomplete development')
        for row in data['table']:tables[row['variant']]=row
    variants={}
    for name in ['v_dev16','sparse_dev16','ranking_baselines_variants','mixed_dev64','gain8_dev64','gain4_dev64']:
        c=load(folder/(name+'.json'))
        for v in c if isinstance(c,list) else c['variants']:variants[v['name']]=v
    # Preserve the main method family and the strongest same-format competitor.
    names=['full8','native4_8','base2_8','cold_meta_all_kv','v_impact_max','v_impact_20',
           'mixed_gradient_max','v_gradient_max','v_gradient_fine_max','v_mse_max','v_spread_max','k_impact_max']
    eligible=[name for name in names if tables[name]['eligible_lowbit_frontier']]
    primary=max(eligible,key=lambda name:(tables[name]['anls'],-tables[name]['budget_bits_including_encoder_constants'],name))
    secondary='v_impact_20'
    if not load(ROOT/'audit/a2/round5_doc64_transport.json')['status']=='PASS':raise ValueError('Doc transport not verified')
    if not load(ROOT/'audit/a2/round5_text64_transport.json')['status']=='PASS':raise ValueError('Text transport not verified')
    selected=[variants[n] for n in names]
    external={}
    for v in selected:
        for key,value in v.items():
            if key.endswith('_path') and isinstance(value,str) and value.startswith(R+'/'):
                path=ROOT/value[len(R)+1:]
                if not path.is_file():raise FileNotFoundError(path)
                external[value]=sha(path)
    # Every confirmation worker sees images/questions, never reference answers.
    base=load(folder/'v_dev16.json')
    configs=[];jobs=[]
    for shard in (0,1):
        config=dict(base,arguments=dict(base['arguments'],samples=R+'/configs/a2/round5_v1/doc_confirm64.json'),
                    variants=selected,start=32*shard,count=32)
        cp=folder/f'confirm_doc_s{shard}.json'
        job={'name':f'round5_confirm_doc_s{shard}_v1','env':{'CUDA_VISIBLE_DEVICES':str(shard)},
             'argv':[P,R+'/code/a2/round5_runner.py','--config',R+'/'+cp.relative_to(ROOT).as_posix(),
                     '--output',R+f'/results/round5_v1/confirm_doc_s{shard}']}
        jp=folder/f'job_confirm_doc_s{shard}.json';configs.extend([(cp,config),(jp,job)]);jobs.append(job['name'])
    transfer_names=list(dict.fromkeys(['full8','native4_8','base2_8','cold_meta_all_kv',primary,secondary,'v_impact_max']))
    for shard in (0,1):
        config=dict(base,arguments=dict(base['arguments'],samples=R+'/configs/a2/round5_v1/text_transfer64.json',turns=1),
                    variants=[variants[n] for n in transfer_names],start=32*shard,count=32)
        cp=folder/f'confirm_text_s{shard}.json'
        queue={'wait_complete':R+f'/results/round5_v1/confirm_doc_s{shard}/complete.json',
               'wait_launch':R+f'/jobs/round5_confirm_doc_s{shard}_v1.launch.json',
               'tasks':[{'name':'transfer32','entrypoint':'round5_runner.py','arguments':['--config',R+'/'+cp.relative_to(ROOT).as_posix()]}]}
        qp=folder/f'confirm_text_queue_s{shard}.json'
        job={'name':f'round5_confirm_text_s{shard}_v1','env':{'CUDA_VISIBLE_DEVICES':str(shard)},
             'argv':[P,R+'/code/a2/run_queue.py','--config',R+'/'+qp.relative_to(ROOT).as_posix(),
                     '--output',R+f'/results/round5_v1/confirm_text_s{shard}']}
        jp=folder/f'job_confirm_text_s{shard}.json';configs.extend([(cp,config),(qp,queue),(jp,job)]);jobs.append(job['name'])
    for path,_ in configs:
        if path.exists() and path.read_text(encoding='utf8')!=json.dumps(dict(configs)[path],indent=2):
            raise RuntimeError('Prepared configuration differs; do not overwrite: '+str(path))
    for path,config in configs:
        if not path.exists():path.write_text(json.dumps(config,indent=2),encoding='utf8')
    inference=['round5_runner.py','factory.py','session_boundary.py','session.py','masked_bank.py',
        'metadata_direction_bank.py','learned_direction_bank.py','sparse_v_residual_bank.py',
        'sparse_v_residual_attention.py','mixed_residual_bank.py','mixed_residual_attention.py',
        'fused_learned_direction.py','fused_masked.py','fast_wht.py',
        'quantizers.py','affine_rate_bank.py']
    report=dict(schema='trajkv_round5_fresh_confirmation_v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        primary=primary,secondary_low_storage=secondary,original_finite_v='v_impact_max',variants=selected,
        primary_selection_rule='highest complete dev64 ANLS among actual packed final <=1.8bit variants; ties lower bits then name; gain4/8 and local DP did not improve the incumbent',
        selected_development_rows={n:tables[n] for n in [primary,secondary,'v_impact_max']},
        budget_bits=1.8,quality_reference='native4_8',quality_margin=.005,
        primary_planned_comparisons=[(primary,'cold_meta_all_kv'),(primary,'base2_8'),(primary,'native4_8')],
        mechanism_planned_comparisons=[('v_impact_max','v_gradient_fine_max'),('v_impact_max','v_mse_max'),
             ('v_impact_max','v_spread_max'),('v_impact_max','k_impact_max'),('v_impact_20','base2_8')],
        statistics='paired source-document bootstrap; primary superiority p-values Holm over three named primary comparisons; 0.5pp point/interval noninferiority reported separately',
        doc_images=64,doc_questions=256,doc_variants=len(selected),text_images=64,text_variants=transfer_names,
        new_doc_outcomes_seen=False,new_text_outcomes_seen=False,methods_change_after_confirmation=False,
        input_sha256={p:sha(ROOT/p) for p in ['configs/a2/round5_v1/doc_confirm64.json','configs/data/round5_doc64_references.jsonl',
            'configs/a2/round5_v1/text_transfer64.json','configs/data/round5_text64_references.json']},
        development_summary_sha256={n:sha(ROOT/'results/round5_v1'/n/'summary.json') for n in summaries},
        calibration_ranking_inputs_sha256={p:sha(ROOT/p) for p in ['configs/a2/next_cal32_independent_questions.json',
            'configs/a2/round5_v1/sparse_rankings.json','results/round5_v1/gradient_queue/calibration16/ranking_scores.json']},
        external_codec_inputs=external,
        inference_source_sha256={**{'code/a2/'+n:sha(ROOT/'code/a2'/n) for n in inference},
            **{n:sha(ROOT/n) for n in ['code/experiments/bank.py','code/experiments/fused_progressive.py','code/bytes.py']}},
        configuration_sha256={p.relative_to(ROOT).as_posix():sha(p) for p,_ in configs},jobs=jobs,
        scope='one new independent Doc source cohort and one new Text image cohort; no full5000x8 restart; one model, frozen FP16 text KV')
    marker.write_text(json.dumps(report,indent=2),encoding='utf8')
    c=connect()
    try:
        for path,_ in configs:upload(c,path,R+'/'+path.relative_to(ROOT).as_posix())
    finally:c.close()
    print(json.dumps({'frozen':True,'primary':primary,'secondary':secondary,'doc_sessions':64*len(selected),
                      'text_sessions':64*len(transfer_names),'jobs':jobs}))


if __name__=='__main__':main()
