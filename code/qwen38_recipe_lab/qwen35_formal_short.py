"""Credit audited80-step history, then use the frozen4K numerical loop to800.

The legacy status enum is retained only for that loop's API. The new fixed
budget/run_class and explicit migration receipt distinguish this formal run.
"""
import argparse,json,os,sys
from pathlib import Path
import torch
from train_qwen35_controls import main as frozen_train,digest
from audit_qwen35_training import audit
from qwen35_long_phase import exact
from qwen35_formal_contract import contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--migration-only',action='store_true');a=p.parse_args()
    assert not a.migration_only or not a.resume
    assert os.getppid()==int(os.environ['QWEN35_CONTROL_GUARD_PID'])
    assert all(os.environ.get(k) is None for k in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
    profile=json.loads(Path(a.profile).read_text());budget,entry=contract(profile,'formal_short');out=Path(profile['out']);torch.set_num_threads(1)
    assert profile['steps']==800 and profile['seq']==4096 and profile['head_chunk']==16 and profile['config']['query_chunk']==128
    assert profile['status']=='resource_probe_candidate' and profile['scope']=='Mac public-family control, not private Table1'
    if not a.resume:
        source=entry['credited_short_checkpoint'];assert digest(source)==entry['credited_short_checkpoint_sha256'] and digest(entry['credited_short_audit'])==entry['credited_short_audit_sha256']
        assert digest(entry['credited_short_profile'])==entry['credited_short_profile_sha256']
        proof=json.loads(Path(entry['credited_short_audit']).read_text());assert proof['status']=='passed_checkpoint_checks' and proof['phase_complete'] and proof['step']==80 and proof['checkpoint_sha256']==entry['credited_short_checkpoint_sha256']
        old=torch.load(source,map_location='cpu',weights_only=True,mmap=True);assert old['profile_sha256']==entry['credited_short_profile_sha256'] and old['step']==80 and old['tokens']==80*4096
        assert all(digest(p)==h for p,h in old['source_hashes'].items())
        for key in ['arm','config','data','batch','seq','head_chunk','seed','device','lr','weight_decay','ns_dtype','clip','balance_weight','mix_probability','schedule','precision','eval_batches']:assert profile[key]==old['profile'][key],key
        out.mkdir(parents=True,exist_ok=False)
        wrapped=dict(old,profile=profile,profile_sha256=digest(a.profile),source_hashes=profile['source_hashes'])
        torch.save(wrapped,out/'migration.tmp');(out/'migration.tmp').replace(out/'last.pt');os.link(out/'last.pt',out/'input-step80.pt')
        loaded=torch.load(out/'input-step80.pt',map_location='cpu',weights_only=True,mmap=True)
        for key in ['model','optimizer','rng','corpora']:exact(loaded[key],old[key])
        migration_audit=audit(a.profile,out/'input-step80.pt');assert migration_audit['step']==80 and not migration_audit['phase_complete']
        Path(out/'migration-audit.json').write_text(json.dumps(migration_audit,indent=2)+'\n')
        Path(out/'migration.json').write_text(json.dumps(dict(source_checkpoint=source,source_checkpoint_sha256=entry['credited_short_checkpoint_sha256'],source_profile_sha256=entry['credited_short_profile_sha256'],destination_profile_sha256=digest(a.profile),formal_budget_sha256=profile['formal_budget_sha256'],credited_steps=80,target_steps=800,credited_tokens=327680,retained_input='input-step80.pt',retained_input_sha256=digest(out/'input-step80.pt'),model_optimizer_rng_corpora_bitwise_after_serialization=True,numerical_loop_unchanged=True,phase_complete=False),indent=2)+'\n')
        del old,wrapped,loaded
    else:
        proof=json.loads(Path(out/'migration.json').read_text());assert proof['destination_profile_sha256']==digest(a.profile) and proof['formal_budget_sha256']==profile['formal_budget_sha256']
        assert digest(out/'input-step80.pt')==proof['retained_input_sha256']
    if a.migration_only:return
    # Same process/parent guard and exact original forward/backward/optimizer code.
    sys.argv=['train_qwen35_controls.py','--profile',a.profile,'--seconds',str(a.seconds),'--resume'];frozen_train()


if __name__=='__main__':main()
