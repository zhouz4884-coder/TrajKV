"""Evaluation views/references of already exposed R7/R8 selection samples.

These use old labels only and never open any R8 fresh confirmation answers.
The grouped calibration32 input is not a TextVQA QA evaluation list; the
independent per-question views here retain official one-question scoring.
"""
import copy
from pathlib import Path
from round8_data_prepare import ROOT,read,sha,new


def main():
    member_path=ROOT/'audit/a2/round7_development_membership_v1.json'
    member=read(member_path); files={}
    for kind in ('doc','text'):
        spec=member['datasets'][kind+'_select16_independent']
        source=ROOT/spec['samples']; reference=ROOT/spec['references']
        assert sha(source)==spec['samples_sha256'] and sha(reference)==spec['references_sha256']
        rows=copy.deepcopy(read(source))
        for row in rows:
            row['source_id']=row['image_id']
            row['round8_analysis_role']='historically_exposed_selection_independent_QA_not_blind'
        target=ROOT/f'configs/a2/round8_v1/{kind}_select16_independent_qa.json'
        new(target,rows)
        files[kind]=dict(samples=str(target.relative_to(ROOT)),samples_sha256=sha(target),
            references=spec['references'],references_sha256=sha(reference),questions=len(rows),
            sources=len({row['source_id'] for row in rows}),input_samples=spec['samples'],input_samples_sha256=sha(source))
    smoke=read(ROOT/'configs/a2/round8_v1/identity_select2_sources_max2q.json')
    grouped=read(ROOT/'configs/a2/round8_v1/calibration_select32.json')
    refpath=ROOT/'configs/data/round8_calibration_select32_references_v1.json'
    refs={row['id']:row for row in read(refpath)}
    by_source={row['source_id']:row for row in grouped}; chosen=[]
    for row in smoke:
        original=by_source[row['source_id']]; ref=copy.deepcopy(refs[original['id']])
        ref['id']=row['id']; ref['references']=ref['references'][:row['turns']]
        assert [str(item['question_id']) for item in ref['references']]==[str(qid) for qid in row['source_question_ids']]
        chosen.append(ref)
    target=ROOT/'configs/data/round8_identity_select2_references_v1.json'; new(target,chosen)
    files['identity_smoke']=dict(references=str(target.relative_to(ROOT)),references_sha256=sha(target),
        parent_references=str(refpath.relative_to(ROOT)),parent_references_sha256=sha(refpath),
        sources=len(smoke),questions=sum(row['turns'] for row in smoke),dataset='doc')
    audit=ROOT/'audit/a2/round8_exposed_evaluation_views_v1.json'
    new(audit,dict(schema='trajkv.round8.exposed-evaluation-views.v1',status='PASS',files=files,
        original_member_sha256=sha(member_path),source_sha256=sha(__file__),
        future_confirmation_answers_opened=False,official_answers_changed=False,
        scope='Existing selection labels only; no independent/blind confirmation claim',GPU_used=False,uploaded=False))
    print(__import__('json').dumps(dict(status='PASS',outputs=files,GPU_used=False)))


if __name__=='__main__': main()
