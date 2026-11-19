"""CPU-only real calTrain32 scheduling checks; no model/labels/network."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random

from round8_continuous import independent_qa
from round8_postcal_schedule import build_schedule,object_sha256,SCHEMA


ROOT=Path(__file__).resolve().parents[2]


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def rejected(fn):
    try:fn()
    except ValueError:return True
    return False


def check(input_path):
    rows=json.loads(input_path.read_text(encoding='utf-8-sig'));before=deepcopy(rows);random_before=random.getstate()
    flat,schedule,meta=build_schedule(rows)
    require(rows==before and random.getstate()==random_before,'Input rows or global RNG were mutated')
    require(len(flat)==70 and len(schedule)==200 and meta['schema']==SCHEMA,'Default dimensions/schema differ')
    require(Counter(x['domain'] for x in schedule)=={'doc':100,'text':100},'Domain step counts not100/100')
    counts=Counter(x['source_id'] for x in schedule)
    require(len(counts)==32 and set(counts.values())=={6,7} and Counter(counts.values())=={6:24,7:8},'Source visit counts not balanced6/7')
    for epoch in range(6):
        part=schedule[epoch*32:(epoch+1)*32]
        require(len({x['source_id'] for x in part})==32 and Counter(x['domain'] for x in part)=={'doc':16,'text':16},'Full epoch loses/repeats a source')
    require(Counter(x['domain'] for x in schedule[192:])=={'doc':4,'text':4}
        and len({x['source_id'] for x in schedule[192:]})==8,'Tail not balanced8distinct sources')
    require(all(x['domain']==('doc' if i%2==0 else 'text') and x['step']==i for i,x in enumerate(schedule)), 'Domains do not interleave or step IDs differ')
    require({x['flat_index'] for x in schedule}==set(range(70)) and meta['unique_questions_scheduled']==70,'Some70Q never scheduled')
    require(meta['schedule_sha256']==object_sha256(schedule),'Schedule SHA differs')
    flat_index=0
    for source_index,row in enumerate(rows):
        for q in range(len(row['questions'])):
            require(flat[flat_index]==independent_qa(row,q),'Frozen independent_qa serialization changed')
            require(flat[flat_index]['turns']==1 and len(flat[flat_index]['questions'])==1,'Independent Q retains history')
            flat_index+=1
        member=[x for x in schedule if x['source_index']==source_index]
        qc=Counter(x['question_index'] for x in member)
        values=[qc[q+1] for q in range(len(row['questions']))]
        require(max(values)-min(values)<=1,'Within-source Q visit imbalance exceeds1')
        for start in range(0,len(member),len(row['questions'])):
            cycle=member[start:start+len(row['questions'])]
            require(len({x['question_index'] for x in cycle})==len(cycle),'Source-local cycle repeats a Q early')
        for item in member:
            qa=flat[item['flat_index']]
            require(item['source_id']==row['source_id'] and item['parent_group_id']==row['id']
                and item['dataset']==row['dataset'] and item['qa_id']==qa['id']
                and item['question_index']==item['question_index_zero_based']+1
                and item['per_step_loss_weight']==1.,'Step identity/reduction contract differs')
    require(abs(sum(meta['flat_question_objective_weights'])-1.)<1e-12,'Exhaustive objective weights do not sum1')
    source_weights=Counter()
    for member,weight in zip(meta['question_membership'],meta['flat_question_objective_weights']):source_weights[member['source_id']]+=weight
    require(all(abs(v-1/32)<1e-12 for v in source_weights.values()),'Exhaustive objective does not weight each source1/32')
    repeated=build_schedule(rows)
    require(repeated==(flat,schedule,meta),'Same seed/input does not reproduce all outputs')
    longer=build_schedule(rows,steps=232)
    # Schedule identity declares total steps, so compare source/Q choices only.
    keys=('source_id','flat_index','question_index','domain','epoch','step')
    require([{k:x[k] for k in keys} for x in longer[1][:200]]==[{k:x[k] for k in keys} for x in schedule], 'Extending steps changes chosen source/Q prefix')
    other=build_schedule(rows,seed=20261119)
    require(other[1]!=schedule and Counter(x['domain'] for x in other[1])=={'doc':100,'text':100}
        and {x['flat_index'] for x in other[1]}==set(range(70)), 'Alternative seed lacks deterministic coverage/balance')
    invalid=[]
    def bad_rows(kind):
        value=deepcopy(rows)
        if kind=='label_payload':value[0]['answers']=['not_read']
        elif kind=='duplicate_source':value[1]['source_id']=value[0]['source_id']
        elif kind=='select_split':value[0]['split']='round8_select_historically_exposed_sources'
        elif kind=='wrong_Q_membership':value[0]['questions'].pop();value[0]['turns']-=1
        elif kind=='turns_mismatch':value[0]['turns']+=1
        return value
    for kind in ('label_payload','duplicate_source','select_split','wrong_Q_membership','turns_mismatch'):
        require(rejected(lambda:build_schedule(bad_rows(kind))), 'Invalid input was accepted:'+kind);invalid.append(kind)
    for steps,seed in ((0,1212),(True,1212),(200,True),(200,'1212')):
        require(rejected(lambda:build_schedule(rows,steps=steps,seed=seed)), 'Bad steps/seed accepted')
    sources={name:sha(Path(__file__).with_name(name)) for name in ('round8_postcal_schedule.py','round8_postcal_schedule_check.py','round8_continuous.py')}
    report=dict(status='PASS',schema='trajkv.round8.postcal-schedule-CPU-check.v1',samples_path=input_path.relative_to(ROOT).as_posix(),
        samples_raw_file_sha256=sha(input_path),source_sha256=sources,seed=1212,steps=200,
        grouped_sources=32,Doc_sources=16,Text_sources=16,Doc_questions=45,Text_questions=25,independent_Q=70,
        domain_steps={'doc':100,'text':100},source_visit_histogram={'6':24,'7':8},full_source_epochs=6,balanced_tail_steps=8,
        all70Q_covered=True,within_source_Q_visit_difference_at_most1=True,
        frozen_independent_QA_serialization_exact=True,input_and_global_RNG_unchanged=True,
        repeated_seed_full_identity=True,extended_schedule_choice_prefix_preserved=True,alternative_seed_20261119_balanced=True,
        exhaustive_source_objective_weights_each1over32=True,per_step_token_mean_loss_weight1=True,
        rejected_invalid_inputs=invalid,schedule_sha256=meta['schedule_sha256'],schedule_id=meta['schedule_id'],
        GT_references_opened=False,select_data_opened=False,GPU_used=False,SSH_used=False,model_forward_executed=False)
    return flat,schedule,meta,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=ROOT/'configs/a2/round8_v1/calibration_train32.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/round8_v1/postcal_schedule_cpu_v1');a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Hide CUDA for this CPU-only check')
    a.output.mkdir(parents=True,exist_ok=False)
    flat,schedule,meta,report=check(a.samples)
    new(a.output/'schedule.json',schedule);new(a.output/'metadata.json',meta);new(a.output/'check.json',report)
    new(a.output/'complete.json',dict(status='PASS',check_sha256=sha(a.output/'check.json'),schedule_file_sha256=sha(a.output/'schedule.json'),
        metadata_file_sha256=sha(a.output/'metadata.json'),GPU_used=False,SSH_used=False,GT_references_opened=False))
    print(json.dumps(dict(status='PASS',sources=32,questions=70,steps=200,domains=[100,100],source_visits=[6,7],
        schedule_sha256=meta['schedule_sha256'],output=str(a.output),GPU_used=False,GT_references_opened=False)))


if __name__=='__main__':main()
