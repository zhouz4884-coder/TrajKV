"""Deterministic source-balanced schedule for answer-free R8 calTrain32.

Only the frozen independent-QA serializer is reused. This module performs no
file/model/network operations and does not inspect labels. The input contract
is the32 original train sources (16Doc/16Text,45+25 questions).
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import random

from round8_continuous import independent_qa,source_identity


SCHEMA='trajkv.round8.postcal-source-balanced-schedule.v1'
DOMAIN_ORDER=('doc','text')
DATASET_DOMAIN={'docvqa_grouped':'doc','textvqa':'text'}
TRAIN_SPLIT='round8_train_historically_exposed_sources'
FORBIDDEN_LABEL_KEYS=frozenset(('answer','answers','references','ground_truth','gt_answer','labels'))


def canonical_json(value):
    """Canonical object bytes; this is not an original file-byte fingerprint."""
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def object_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _require(condition,message):
    if not condition:raise ValueError(message)


def _validate(rows):
    _require(isinstance(rows,(list,tuple)) and len(rows)==32,'Exactly32 grouped calTrain sources required')
    sources=[];ids=[];images=[];datasets=Counter();questions=Counter()
    for row in rows:
        _require(isinstance(row,dict),'Each source must be a mapping')
        _require(not FORBIDDEN_LABEL_KEYS.intersection(row),'Labels/reference payloads are forbidden in schedule inputs')
        _require(row.get('dataset') in DATASET_DOMAIN,'Only DocVQA/TextVQA train domains are supported')
        _require(row.get('split')==TRAIN_SPLIT and row.get('calibration_only') is True,'Only the declared calTrain split is eligible')
        sid=source_identity(row);image=row.get('image_id');gid=row.get('id');q=row.get('questions')
        _require(all(isinstance(value,str) and value for value in (sid,image,gid)),'Nonempty source/image/group IDs required')
        _require(isinstance(q,list) and q and all(isinstance(value,str) and value.strip() for value in q),'Nonempty question strings required')
        _require(type(row.get('turns')) is int and row['turns']==len(q),'Grouped turns must equal the declared question count')
        _require(row.get('session_id',gid)==gid,'Group session_id differs from id')
        sources.append(sid);images.append(image);ids.append(gid)
        domain=DATASET_DOMAIN[row['dataset']];datasets[domain]+=1;questions[domain]+=len(q)
    _require(len(set(sources))==len(set(images))==len(set(ids))==32,'Repeated source/image/group IDs are forbidden')
    _require(dict(datasets)=={'doc':16,'text':16},'Exactly16 original sources per domain required')
    _require(dict(questions)=={'doc':45,'text':25},'Preserve the exact45Doc+25Text question membership')


def build_schedule(grouped_rows,steps=200,seed=1212):
    """Return ``(flat_QA_rows, schedule, metadata)`` without mutating inputs.

    Each source epoch contains each32 sources once with alternating Doc/Text.
    Source order is shuffled independently per domain/epoch. Each source has
    its own shuffled question cycles; a question repeats only after its full
    source-local cycle. For200 steps: six full source epochs plus an8-step
    balanced tail,100 steps/domain and6/7 visits/source; all70Q are covered.

    One optimization step uses its selected Q's token-mean KL with weight1.
    ``flat_question_objective_weights`` instead weights exhaustive70Q summary
    evaluation:1/(2*16*number_of_questions_for_its_source). Applying it again
    to the sampled step loss would double-weight the question sampling.

    Object SHA values below hash canonical JSON, not original file bytes.
    The caller must separately pin the actual samples-file byte SHA.
    """
    _require(type(steps) is int and steps>0,'steps must be a positive integer')
    _require(type(seed) is int,'seed must be an integer')
    _validate(grouped_rows)
    rows=deepcopy(list(grouped_rows));rng=random.Random(seed)
    flat=[];source_members=[];question_members=[];by_domain={d:[] for d in DOMAIN_ORDER};source_flats={}
    weights=[]
    for source_index,row in enumerate(rows):
        sid=source_identity(row);domain=DATASET_DOMAIN[row['dataset']];by_domain[domain].append(source_index)
        source_flats[source_index]=[]
        source_members.append(dict(source_index=source_index,source_id=sid,dataset=row['dataset'],domain=domain,
            image_id=row['image_id'],parent_group_id=row['id'],question_count=len(row['questions'])))
        for question_zero in range(len(row['questions'])):
            qa=independent_qa(row,question_zero);index=len(flat);flat.append(qa);source_flats[source_index].append(index)
            question_members.append(dict(flat_index=index,qa_id=qa['id'],source_index=source_index,source_id=sid,
                parent_group_id=row['id'],dataset=row['dataset'],domain=domain,question_index=question_zero+1,
                question_index_zero_based=question_zero,
                question_text_utf8_sha256=hashlib.sha256(row['questions'][question_zero].encode('utf-8')).hexdigest(),
                independent_qa_canonical_sha256=object_sha256(qa)))
            weights.append(1./(32*len(row['questions'])))
    source_sha=object_sha256(source_members);question_sha=object_sha256(question_members)
    identity=dict(schema=SCHEMA,steps=steps,seed=seed,source_membership_sha256=source_sha,question_membership_sha256=question_sha)
    schedule_id='r8-postcal-sourcebalanced-v1-'+object_sha256(identity)[:24]
    # Keep question RNG independent of source permutations: extending steps
    # leaves both the source-order and question-order schedule prefix intact.
    q_rng={i:random.Random(object_sha256(dict(seed=seed,source_id=source_identity(row),purpose='question-cycle-v1')))
        for i,row in enumerate(rows)}
    q_cycles={i:[] for i in range(32)};source_visits=Counter();question_visits=Counter();schedule=[]
    epoch=0
    while len(schedule)<steps:
        orders={domain:list(by_domain[domain]) for domain in DOMAIN_ORDER}
        for domain in DOMAIN_ORDER:rng.shuffle(orders[domain])
        for rank in range(16):
            for domain in DOMAIN_ORDER:
                if len(schedule)==steps:break
                source_index=orders[domain][rank];row=rows[source_index]
                if not q_cycles[source_index]:
                    cycle=list(source_flats[source_index]);q_rng[source_index].shuffle(cycle);q_cycles[source_index]=cycle
                flat_index=q_cycles[source_index].pop(0);member=question_members[flat_index]
                step=dict(schema=SCHEMA,schedule_id=schedule_id,step=len(schedule),epoch=epoch,
                    epoch_position=2*rank+DOMAIN_ORDER.index(domain),flat_index=flat_index,
                    source_index=source_index,source_id=member['source_id'],parent_group_id=member['parent_group_id'],
                    dataset=member['dataset'],domain=domain,question_index=member['question_index'],
                    question_index_zero_based=member['question_index_zero_based'],qa_id=member['qa_id'],
                    source_visit_index=source_visits[source_index],question_visit_index=question_visits[flat_index],
                    per_step_loss_reduction='mean over this Q teacher target tokens',per_step_loss_weight=1.)
                schedule.append(step);source_visits[source_index]+=1;question_visits[flat_index]+=1
        epoch+=1
    source_counts=[dict(source_index=i,source_id=source_identity(row),dataset=row['dataset'],domain=DATASET_DOMAIN[row['dataset']],
        visits=source_visits[i],question_visits=[question_visits[j] for j in source_flats[i]]) for i,row in enumerate(rows)]
    domain_counts=dict(Counter(step['domain'] for step in schedule))
    metadata=dict(identity,schedule_id=schedule_id,source_count=32,question_count=70,
        sources_per_domain={'doc':16,'text':16},questions_per_domain={'doc':45,'text':25},domain_step_counts=domain_counts,
        grouped_rows_canonical_sha256=object_sha256(rows),flat_qa_rows_canonical_sha256=object_sha256(flat),
        schedule_sha256=object_sha256(schedule),hash_serialization='UTF8 canonical JSON:sort_keys,separators comma/colon,ensure_ascii=False,allow_nan=False',
        raw_samples_file_sha256_included=False,
        raw_samples_file_binding='caller must separately bind exact file bytes; no file I/O is performed here',
        source_membership=source_members,question_membership=question_members,
        flat_question_objective_weights=weights,
        objective='(1/2) sum_domain (1/16) sum_source (1/Q_source) sum_Q (1/T_Q) sum_teacher_tokens KL(Full||postcal)',
        objective_scope='Expected domain/source/question balanced token-mean KL; finite200 schedule gives6/7 visits/source, not exact equal counts',
        exhaustive_objective_weights_scope='Use flat_question_objective_weights only for enumerated70Q aggregates; a scheduled single-Q step already samples this objective and has weight1',
        source_visit_counts=source_counts,unique_questions_scheduled=len(question_visits),
        full_source_epochs=steps//32,tail_steps=steps%32,domain_order=list(DOMAIN_ORDER),
        schedule_policy='Per-epoch shuffled domain-specific source permutations interleaved Doc/Text; independent seeded shuffled source-local Q cycles',
        reference_data_used=False,model_outputs_used=False,select_data_used=False)
    return flat,schedule,metadata
