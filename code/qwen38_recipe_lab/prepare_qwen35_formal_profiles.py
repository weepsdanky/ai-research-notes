"""Declare a new formal profile without modifying any historical profile/weights."""
import argparse,json
from datetime import datetime,timedelta
from pathlib import Path
from qwen35_formal_contract import BUDGET,contract
from run_dense_text_guarded import load,digest,absent


def token_plan(profile,budget,stage):
    """Replace inherited feasibility prose/counters before declaring a new file."""
    phase=budget[stage]
    profile['planned_tokens']=phase['total_steps']*phase['seq']*phase['batch']
    if stage=='long':
        profile['status']='formal_phase_predeclared'
        short=budget['short'];parent_tokens=short['total_steps']*short['seq']*short['batch']
        profile['planned_total_tokens']=parent_tokens+profile['planned_tokens']
        profile['eval_every']=phase['total_steps']
        profile['validation_execution']='separate_after_long_phase_at32k_and4k_with_fixed_batches'
        profile['phase_boundary']=f"Preserve actual completed{short['total_steps']}x{short['seq']} model/optimizer/private RNG exactly; immutable initial.pt; cumulative{parent_tokens}+{profile['planned_tokens']} tokens. Feasibility long updates are excluded."
        profile['confirmatory_common_engine']='Same query32/head8 streamed CE and whole-layer rematerialization; actual formal800-step parent and20 new32K updates, with no feasibility-step credit.'
        profile['memory_variant_scope']='Unchanged numerical operators and full softmax with whole-layer rematerialization and streamed head gradients; formal lineage/budget are independently audited.'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['full','swa','gdn'],required=True);p.add_argument('--stage',choices=['short','long'],required=True);p.add_argument('--out',required=True);p.add_argument('--run-dir');p.add_argument('--parent-profile');p.add_argument('--parent-state');p.add_argument('--resource-samples');p.add_argument('--deadline');a=p.parse_args()
    path=Path(a.out);assert not path.exists();budget=load(BUDGET);entry=next(row for row in budget['arms'] if row['arm']==a.arm)
    shared=['qwen35_formal_contract.py','qwen35_formal_review.py','run_qwen35_formal_guarded.py',BUDGET]
    if a.stage=='short':
        assert not a.parent_profile and not a.parent_state and not a.resource_samples and not a.deadline
        assert digest(entry['credited_short_profile'])==entry['credited_short_profile_sha256'] and digest(entry['credited_short_audit'])==entry['credited_short_audit_sha256']
        profile=load(entry['credited_short_profile']);profile['steps']=800;profile['eval_every']=100;profile['stage']='formal_short'
        profile['out']=a.run_dir or f'runs/qwen35-formal-{a.arm}-4k-cpu'
        profile['training_execution']='frozen_4k_loop_with_explicit_credited80_metadata_migration'
        files=shared+['qwen35_formal_short.py']
    else:
        assert a.parent_profile and a.parent_state and a.resource_samples and a.deadline
        parent=load(a.parent_profile);contract(parent,'formal_short');state=load(a.parent_state);audit=load(state['audit_file'])
        deadline=datetime.fromisoformat(a.deadline);now=datetime.now().astimezone();assert deadline.tzinfo and now+timedelta(seconds=900)<deadline<=now+timedelta(hours=2)
        assert deadline<=datetime.fromisoformat(state['deadline']) or now>=datetime.fromisoformat(state['deadline'])
        samples=load(a.resource_samples);assert len(samples)==12 and all(r['pressure']==1 and r['free']>=55 for r in samples)
        stamps=[datetime.fromisoformat(r['at']) for r in samples];assert all(t.tzinfo for t in stamps) and all(4<=(b-a).total_seconds()<=15 for a,b in zip(stamps,stamps[1:]))
        assert 0<=(now-stamps[-1]).total_seconds()<=120
        assert parent['arm']==a.arm and state['status']=='saved_and_audited' and not state['signals'] and state['returncode']==0
        assert state['step']==800 and state['phase_complete'] and digest(a.parent_profile)==state['profile_sha256']==audit['profile_sha256']
        assert audit['status']=='passed_checkpoint_checks' and audit['phase_complete'] and audit['step']==800
        assert digest(state['checkpoint'])==state['checkpoint_sha256']==audit['checkpoint_sha256'] and digest(state['audit_file'])==state['audit_sha256']
        for pid in [state['pid'],state['child_pid']]:absent(pid)
        profile=load(f'profiles/qwen35_{a.arm}_32k_tiled_common_cpu.json');profile['steps']=20;profile['stage']='formal_long';profile['out']=a.run_dir or f'runs/qwen35-formal-{a.arm}-32k-cpu'
        profile['parent_profile']=a.parent_profile;profile['parent_checkpoint']=state['checkpoint'];profile['parent_checkpoint_sha256']=state['checkpoint_sha256'];profile['parent_audit']=state['audit_file'];profile['parent_audit_sha256']=state['audit_sha256']
        profile['training_execution']='formal_tiled_one_update_separate_validation';profile['runtime_tiling']=dict(query_chunk=32,head_chunk=8)
        profile['resource_readmission']=dict(samples_file=a.resource_samples,samples_sha256=digest(a.resource_samples),deadline=deadline.isoformat())
        profile['source_hashes'].update(parent['source_hashes']);files=shared+['qwen35_formal_long.py','audit_qwen35_formal_long.py']
    token_plan(profile,budget,a.stage)
    assert not Path(profile['out']).exists(),'Never overwrite an existing formal namespace'
    profile['run_class']='formal_predeclared_continuation';profile['formal_budget_manifest']=BUDGET;profile['formal_budget_sha256']=digest(BUDGET)
    profile['legacy_status_compatibility']='The frozen4K numerical loop requires resource_probe_candidate; fixed formal_budget SHA/run_class/steps define this new formal trajectory.' if a.stage=='short' else 'Formal long uses its own declared phase status, actual800-step parent and20-step target; no feasibility update credit.'
    profile['source_hashes'].update({name:digest(name) for name in files})
    profile['profile_builder_provenance']=dict(file=__file__,sha256=digest(__file__),runtime_dependency=False)
    contract(profile,profile['stage'])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(profile,f,indent=2)
    print(path,digest(path))


if __name__=='__main__':main()
