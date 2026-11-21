"""CPU-only exact24 matched-point owner/protocol proof; no GPU/transport."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path('${PROJECT_ROOT}');E=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'code')];sys.dont_write_bytecode=True
import round9_formal_public_contract_v6 as v
from verify_public_round9_formal_v6 import physical_complete_bytes
B=Path('${EVIDENCE_ROOT}/results/publication/R9_formal4_selected_bundle_v6');bundle=v.load(B/'BUNDLE.json')
common=v.load(v.same500.COMMON);proofs=[];started=time.time()
for f in [bundle['primary_contract'],*bundle['auxiliary_contracts']]:
    c=v.load(f);rows=v.load(c['matrix'])['rows']
    for row in rows:
        raw,scored=v.resolve(row['view']),v.resolve(row['scored']);m=v.load(raw/'manifest.json');s=v.load(scored/'summary.json');complete=v.load(raw/'complete.json')
        assert m['expected_QA']==s['expected_run_questions']==500 and complete['counts']=={'PASS':500} and s['status']=='PASS_SCORED_COMPLETE'
        ds=row['dataset'];context=common['common_contracts_by_dataset'][ds]['context']
        assert v.same500.frozen.native_context(m)=={k:context[k]for k in v.same500.frozen.native_context(m)} and m['model_spec']==context['model_spec']
        records={}
        with(scored/'scored_records.jsonl').open(encoding='utf-8')as stream:
            for line in stream:
                r=json.loads(line);assert r['id']not in records;records[r['id']]=r
        amounts=[];bits=[];identities={};full_sha=hashlib.sha256();over13=over18=0
        with(raw/'records.jsonl').open('rb')as stream:
            for number,line in enumerate(stream,1):
                full_sha.update(line);r=json.loads(line);sr=records[r['id']]
                assert r['status']=='PASS' and sr['raw_status']=='PASS' and sr['raw_jsonl_line_number']==number and sr['raw_jsonl_exact_line_sha256']==hashlib.sha256(line).hexdigest()
                actual=physical_complete_bytes(r,m['model_spec']);n=r['actual_visual_token_count'];den=r['full_kv_byte_denominator']
                if c['family']==v.FAMILY:v.original_r9.verify_brpe_row(r,m['model_spec'],'BRPE_Drop4')
                amounts.append(actual);bits.append(16*actual/den);over13+=160*actual>13*den;over18+=80*actual>9*den
                assert abs(bits[-1]-r['actual_effective_visual_bits'])<1e-12
                identities[r['id']]=v.same500.frozen.input_identity(r)
        assert len(identities)==len(amounts)==500 and full_sha.hexdigest()==complete['records_jsonl_sha256']==s['integrity']['raw_files']['records.jsonl']
        assert v.same500.frozen.canonical(identities)==common['parent_full500_alignment'][ds]['per_ID_identity_fingerprint']
        proofs.append({'cell':row['cell'],'dataset':ds,'reference_key':row.get('reference_key'),'primary_fresh':c['family']==v.FAMILY,
            'actual_QA':500,'complete_owner_tables_reconstructed':500,'mean_complete_bytes':sum(amounts)/500,'max_complete_bytes':max(amounts),
            'mean_bits':sum(bits)/500,'max_bits':max(bits),'over_strict13':over13,'over_strict18':over18,
            'raw_file_sha256':s['integrity']['raw_files'],'score_summary_sha256':v.sha(scored/'summary.json'),
            'full_same500_input_protocol_model_runtime_verified':True,'source_admission_not_replaced_by_this_owner_scan':True})
        print(json.dumps({'cell':row['cell'],'owner_tables':500,'complete':len(proofs)}),flush=True)
assert len(proofs)==24 and sum(p['actual_QA']for p in proofs if p['primary_fresh'])==2000
v.write(E/'PRIVATE24_PHYSICAL_AND_PROTOCOL.json',{'status':'PASS_ACTUAL24_POINTS_12000_OWNER_TABLES_AND_SAME500_PROTOCOL',
    'actual_primary_fresh_answers':2000,'auxiliary_reused_view_answers':10000,'auxiliary_new_GPU_QA':0,'points':proofs,
    'source':v.sha(__file__),'common_contract_sha256':v.sha(v.same500.COMMON),'elapsed_seconds':time.time()-started,
    'global_old47_not_scanned':True,'whole_export_or_cold_performed':False,'new_GPU_or_remote_or_Git':False})
