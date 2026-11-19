"""CPU config/real-marker/snapshot-launch controls; no network or workers."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

from round8_data_prepare import ROOT,read,sha,new
from round8_quality_owner import local_path
from analyze_round8_final_quality import completion_gate


def require(value,message):
    if not value:raise RuntimeError(message)


def actual_control_body():
    tree=ast.parse((ROOT/'code/launch_remote_data_disk.py').read_text(encoding='utf-8'))
    return next(node.value.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='code' for target in node.targets)
        and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))


def launcher_fixture(output,job,index):
    """Execute exact frozen embedded control body with only Popen stubbed."""
    folder=output/('launcher_fixture_'+str(index));sr=folder/'sourceRoot';(sr/'code/a2').mkdir(parents=True)
    entry=Path(job['argv'][1]).name;source=sr/'code/a2'/entry;source.write_text('# CPU launch fixture only\n',encoding='utf-8')
    fixture=copy.deepcopy(job);fixture['cwd']=str(sr)
    # Match the exact root prefix plus POSIX separator used by the Linux
    # control body while its Path operations are exercised on Windows.
    fixture['argv'][1]=str(sr/'code')+'/a2/'+entry
    for key,leaf in [('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda')]:
        fixture['env'][key]=str(folder/'globalData/cache'/leaf)
    jobs=sr/'jobs';jobs.mkdir();(sr/'logs').mkdir();cp=jobs/(job['name']+'.json');new(cp,fixture)
    calls=[];fake=ModuleType('subprocess')
    def popen(argv,**kwargs):
        calls.append(dict(argv=argv,cwd=kwargs['cwd'],env=kwargs['env']))
        return SimpleNamespace(pid=987000+index)
    fake.Popen=popen;fake.STDOUT=-2
    stream=io.StringIO()
    with patch.dict(sys.modules,{'subprocess':fake}),patch.object(sys,'argv',['control-fixture',str(cp)]),redirect_stdout(stream):
        exec(compile(actual_control_body(),str(ROOT/'code/launch_remote_data_disk.py')+':embedded','exec'),{})
    receipt=json.loads(stream.getvalue());expected=sr/'jobs'/(job['name']+'.source')/'code/a2'/entry
    require(len(calls)==1 and calls[0]['argv'][1]==str(expected) and expected.is_file(),'Frozen launcher did not execute copied source entry')
    require(sha(expected)==sha(source) and receipt['argv_executed']==calls[0]['argv'],'Snapshot/source/Popen identity mismatch')
    require(calls[0]['argv'][0]==job['argv'][0] and calls[0]['argv'][2:]==job['argv'][2:] and calls[0]['cwd']==str(sr),
        'Worker runtime/global config/output/cwd unexpectedly rewritten')
    # Independently execute the dedicated wrapper with fake launcher/remote;
    # no import of credential routing or actual launcher main occurs.
    wrapper=ROOT/'code/launch_round8_data_disk.py';launcher=ModuleType('launch_remote_data_disk');remote=ModuleType('remote')
    remote.REMOTE_ROOT=job['round8_source_root'].split('/stages/',1)[0];launcher.REMOTE_ROOT=remote.REMOTE_ROOT;launcher.PYTHON='NOT_A_REAL_WORKER'
    wrapper_calls=[];launcher.main=lambda:wrapper_calls.append(dict(root=launcher.REMOTE_ROOT,control_python=launcher.PYTHON))
    local=output/('wrapper_job_'+str(index)+'.json');new(local,job)
    with patch.dict(sys.modules,{'launch_remote_data_disk':launcher,'remote':remote}),patch.object(sys,'argv',[str(wrapper),str(local)]):
        env={'__name__':'CPU_wrapper_fixture'};exec(compile(wrapper.read_text(encoding='utf-8'),str(wrapper),'exec'),env);env['main']()
    require(wrapper_calls==[dict(root=job['round8_source_root'],control_python='/usr/bin/python3')],
        'Dedicated wrapper did not relocate launcher root before launch')
    return dict(status='PASS',actual_frozen_embedded_body_executed=True,dedicated_wrapper_executed_with_stubs=True,
        Popen_stub_calls=1,real_worker_started=False,remote_connection_imported=False,
        entry_inside_job_snapshot=True,copied_source_exact=True,global_config_output_and_worker_argv_preserved=True,
        expected_real_argv_executed=job['expected_launch_receipt']['argv_executed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rp=ROOT/'audit/a2/round8_natural_ray_mean_v1_prepared.json';r=read(rp);plans=[];launches=[];baselines={};hashes={str(rp.relative_to(ROOT)):sha(rp)}
    require(r['total_expected_records']==256 and len(r['outputs'])==4,'Incomplete four-job preparation')
    for index,spec in enumerate(r['outputs']):
        cp,jp=ROOT/spec['config'],ROOT/spec['job'];c,j=read(cp),read(jp)
        require(sha(cp)==spec['config_sha256'] and sha(jp)==spec['job_sha256'],'Prepared config/job changed')
        backend=importlib.import_module(c['backend_module']);declarations=backend.validate_config(c)
        require(backend.source_fingerprints()==r['source_sha256'][spec['family']] and len(declarations)==1,'Backend source/declarations differ')
        require(c['arguments']['max_patches']==8 and c['variants'][0]['cap']==8 and c['count']==64 and c['start']==0
            and c['arguments']['turns']==1,'Explicit cap8/single first-QA contract differs')
        samples=local_path(c['arguments']['samples']);rows=read(samples)
        require(sha(samples)==spec['samples_sha256'] and len(rows)==len({v['id'] for v in rows})==len({v['image_id'] for v in rows})==64,
            'Exact64 sample identities differ')
        require(all(row.get('session_id')==row['id'] and len(row['questions'])==row['turns']==1
            and not any(key in row for key in ('answer','answers','references')) for row in rows),'Question or answer-free schema differs')
        original=ROOT/('results/round8_v1/natural_dev64_firstq3_v1_'+spec['dataset']);manifest=read(original/'manifest.json');complete=read(original/'complete.json')
        require((original/'complete_round8.json').is_file() and complete['status']=='COMPLETE' and complete['records']==complete['expected']==192,
            'Original three-arm producer incomplete')
        require(manifest['selected_ids']==[row['id'] for row in rows] and manifest['samples_sha256']==sha(samples),'Original source order/input differs')
        oldnames=[v['name'] for v in manifest['config']['variants']]
        require(oldnames==['full8','r7_frozen_joint43_control','r8_oldbook_binaryV_jointgain'],'Original baseline names differ')
        expected_names=oldnames+[c['variants'][0]['name']]
        require(len(set(expected_names))==4 and spec['analysis_argv'][spec['analysis_argv'].index('--expected-variants')+1]=='4'
            and '--variants' not in spec['analysis_argv'],'Each candidate must merge separately with every original arm')
        sr=j['round8_source_root'];receipt=j['expected_launch_receipt'];entry=c['backend_module']+'.py'
        require(j['required_launcher']=='code/launch_round8_data_disk.py' and j['cwd']==sr
            and receipt['source_snapshot']==sr+'/jobs/'+j['name']+'.source'
            and receipt['argv_executed'][1]==receipt['source_snapshot']+'/code/a2/'+entry
            and receipt['job'].startswith(sr+'/jobs/') and receipt['log'].startswith(sr+'/logs/'), 'True sourceRoot snapshot contract differs')
        command=[sys.executable,str(ROOT/'code/a2'/entry),'--config',str(cp),'--output',str(a.output/('unused_'+str(index))),'--plan-only']
        done=subprocess.run(command,text=True,capture_output=True,check=True);plan=json.loads(done.stdout)
        require(plan['status']=='PLAN_ONLY' and plan['GPU_executed'] is False and len(plan['variants'])==1,'Actual plan-only runner did not preserve singleton candidate')
        plans.append(dict(family=spec['family'],dataset=spec['dataset'],expected_records=64,variant=c['variants'][0]['name'],
            candidate_validate_and_real_CLI_plan_only='PASS',separate_merged4_names=expected_names))
        launches.append(launcher_fixture(a.output,j,index));hashes[spec['config']]=sha(cp);hashes[spec['job']]=sha(jp)
        baselines[spec['dataset']]=dict(records=192,three_arm_names=oldnames,cap_actual_per_variant=8,exact_samples_and_ID_order=True)
    # These are existing fully accepted real markers, not synthesized future QA.
    marker_controls=[]
    for folder in ('final_ray_select5_v1_doc','final_ray_select5_v1_text','mean_select2_v2_doc','mean_select2_v2_text'):
        path=ROOT/'results/round8_v1'/folder;gate=completion_gate(path)
        require(gate and gate['status']=='PASS','Existing real marker schema control failed')
        marker_controls.append(dict(run=folder,final_marker=gate['final_marker'],status='PASS',marker_sha256=gate['final_marker_sha256']))
    result=dict(status='PASS',schema='trajkv.round8.natural-final-ray-mean-CPU-check.v1',config_plans=plans,baseline_controls=baselines,
        launcher_controls=launches,existing_real_marker_controls=marker_controls,source_sha256=r['source_sha256'],files_sha256=hashes,
        checker_sha256=sha(__file__),dedicated_launcher_sha256=sha(ROOT/'code/launch_round8_data_disk.py'),
        generic_embedded_control_sha256=sha(ROOT/'code/launch_remote_data_disk.py'),
        future_acceptance='Only dedicated final markers and independently verified canonical/source/config/input/64record graphs permit scoring; actual launch receipt/PID cmdline must match snapshot',
        total_future_records=256,GT_references_parsed=False,GPU_used=False,SSH_used=False,model_forward_executed=False,real_worker_started=False,
        scope='CPU local plans, exact launcher control body with Popen/connection stubs, and existing real final-marker schema checks only; new natural jobs not executed')
    new(a.output/'check.json',result);new(a.output/'complete.json',dict(status='PASS',GPU_used=False,SSH_used=False,check_sha256=sha(a.output/'check.json')))
    print(json.dumps(dict(status='PASS',config_plans=4,exact_launcher_fixture_cases=4,existing_real_marker_controls=4,future_records=256,GPU_used=False,SSH_used=False)),flush=True)


if __name__=='__main__':main()
