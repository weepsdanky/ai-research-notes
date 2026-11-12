"""One exact full-vocabulary update per process; validation runs separately.

Explicit profile migration preserves accepted model/optimizer/private RNG state.
All original32K failed attempts and their source packages remain untouched.
"""
import argparse,json,os,resource,time,mmap
from pathlib import Path
import torch
from qwen35_controls import ControlConfig
from qwen35_layer_checkpoint import LayerCheckpointControl
from optim import RecipeOptimizer,ownership
from train_qwen35_controls import Corpus,balance_loss,digest
from qwen35_long_phase import exact
from audit_qwen35_training import finite
from streamed_lm_backward import streamed_lm_backward


def parent_state(profile):
    assert digest(profile['parent_checkpoint'])==profile['parent_checkpoint_sha256'] and digest(profile['parent_audit'])==profile['parent_audit_sha256']
    audit=json.loads(Path(profile['parent_audit']).read_text());assert audit['phase_complete'] and audit['status']=='passed_checkpoint_checks' and audit['checkpoint_sha256']==profile['parent_checkpoint_sha256']
    parent=torch.load(profile['parent_checkpoint'],map_location='cpu',weights_only=True,mmap=True);short=parent['profile']
    assert parent['profile_sha256']==digest(profile['parent_profile'])==audit['profile_sha256']
    assert parent['step']==short['steps']==80 and short['seq']==4096 and parent['tokens']==80*4096*short['batch']
    for key in ['arm','config','data','batch','seed','device','lr','weight_decay','ns_dtype','clip','balance_weight','mix_probability','schedule','precision','head_chunk']:assert profile[key]==short[key]
    assert all(digest(p)==sha for p,sha in parent['source_hashes'].items())
    return parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert os.getppid()==int(os.environ['QWEN35_CONTROL_GUARD_PID']) and 1<=a.seconds<=1140
    profile=json.loads(Path(a.profile).read_text());assert profile['phase']=='32k_continuation' and profile['seq']==32768 and profile['steps']==2 and profile['device']=='cpu'
    assert profile['training_execution']=='streamed_backward_one_update_separate_validation'
    assert {'qwen35_streamed_update.py','streamed_lm_backward.py','run_qwen35_streamed_guarded.py'}<=profile['source_hashes'].keys()
    assert all(digest(p)==sha for p,sha in profile['source_hashes'].items())
    assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
    torch.set_num_threads(1);torch.manual_seed(profile['seed']);parent=parent_state(profile);out=Path(profile['out'])
    assert torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE,'Optimizer buffers require private copy-on-write mappings'
    if not a.resume:
        admission=profile['resource_readmission'];samples=json.loads(Path(admission['samples_file']).read_text());assert digest(admission['samples_file'])==admission['samples_sha256'] and len(samples)==12 and all(s['pressure']==1 and s['free']>=55 for s in samples)
        from datetime import datetime
        assert 0<=(datetime.now().astimezone()-datetime.fromisoformat(samples[-1]['at'])).total_seconds()<=120
        out.mkdir(parents=True,exist_ok=False)
    corpora=[Corpus(row['directory'],row['meta_sha256'],32768) for row in profile['data']];assert [c.meta for c in corpora]==parent['corpora']
    def wrapped(values,step):
        return dict(profile=profile,profile_sha256=digest(a.profile),source_hashes=profile['source_hashes'],config=profile['config'],arm=profile['arm'],phase=profile['phase'],model=values['model'],optimizer=values['optimizer'],rng=values['rng'],corpora=parent['corpora'],step=step,phase_tokens=step*32768*profile['batch'],tokens=parent['tokens']+step*32768*profile['batch'],optimizer_steps_offset=80,parent_training_tokens=parent['tokens'],parent_checkpoint_sha256=profile['parent_checkpoint_sha256'])
    if a.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True,mmap=True);assert saved['profile_sha256']==digest(a.profile) and saved['source_hashes']==profile['source_hashes'];step=saved['step']
    elif profile.get('migration_checkpoint'):
        assert digest(profile['migration_checkpoint'])==profile['migration_checkpoint_sha256'] and digest(profile['migration_audit'])==profile['migration_audit_sha256']
        old_audit=json.loads(Path(profile['migration_audit']).read_text());assert old_audit['status']=='passed_checkpoint_checks' and old_audit['checkpoint_sha256']==profile['migration_checkpoint_sha256'] and not old_audit['phase_complete']
        saved=torch.load(profile['migration_checkpoint'],map_location='cpu',weights_only=True,mmap=True);step=saved['step'];assert step==old_audit['step']==1
        assert old_audit['profile_sha256']==saved['profile_sha256']==digest(profile['migration_profile'])
        assert saved['parent_checkpoint_sha256']==profile['parent_checkpoint_sha256']
        for key in ['arm','config','data','batch','seed','device','lr','weight_decay','ns_dtype','clip','balance_weight','mix_probability','schedule','precision','head_chunk','seq','steps']:assert profile[key]==saved['profile'][key]
        assert all(digest(p)==sha for p,sha in saved['source_hashes'].items())
        cold=torch.load(old_audit['initial_checkpoint'],map_location='cpu',weights_only=True,mmap=True);assert digest(old_audit['initial_checkpoint'])==old_audit['initial_checkpoint_sha256']
        for key in ['model','optimizer','rng','corpora']:exact(cold[key],parent[key])
        torch.save(wrapped(cold,0),out/'initial.pt');del cold
        torch.save(wrapped(saved,step),out/'last.pt');os.link(out/'last.pt',out/'migrated-input.pt')
        Path(out/'migration.json').write_text(json.dumps(dict(source_checkpoint=profile['migration_checkpoint'],source_checkpoint_sha256=profile['migration_checkpoint_sha256'],source_audit_sha256=profile['migration_audit_sha256'],source_profile_sha256=saved['profile_sha256'],destination_profile_sha256=digest(a.profile),accepted_step_preserved=step,model_optimizer_rng_bitwise_checked=True,scope='Metadata/profile migration only before the next update; immutable old checkpoint retained'),indent=2)+'\n')
    else:
        saved=parent;step=0;torch.save(wrapped(parent,0),out/'initial.pt');os.link(out/'initial.pt',out/'last.pt')
    assert 0<=step<2
    model=LayerCheckpointControl(ControlConfig(**profile['config']),profile['arm'],seed=profile['seed']).train();model.load_state_dict(saved['model'],strict=True)
    assert model.head.weight.data_ptr()!=model.embedding.weight.data_ptr(),'The streamed adjoint assumes the published untied head'
    optimizer=RecipeOptimizer(ownership(model),profile['lr'],weight_decay=profile['weight_decay'],ns_dtype=profile['ns_dtype']);optimizer.load_state_dict(saved['optimizer'])
    rng=torch.Generator();rng.set_state(saved['rng'])
    exact(model.state_dict(),saved['model']);exact(optimizer.state_dict(),saved['optimizer']);exact(rng.get_state(),saved['rng'])
    initial_tokens=parent['tokens'];del saved,parent
    optimizer.zero_grad(set_to_none=True);tick=time.monotonic();index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
    x,y=corpora[index].batch('train',profile['batch'],32768,rng,'cpu');hidden,routers=model.hidden(x);balance=balance_loss(routers,model.config.experts,model.config.top_k);assert torch.isfinite(balance).item()
    lm,adjoint=streamed_lm_backward(hidden,model.head.weight,y,profile['head_chunk']);torch.autograd.backward((hidden,balance),(adjoint,balance.new_tensor(profile['balance_weight'])))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),profile['clip'],error_if_nonfinite=True,foreach=False);optimizer.step();step+=1
    values=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state());state=dict(profile=profile,profile_sha256=digest(a.profile),source_hashes=profile['source_hashes'],config=profile['config'],arm=profile['arm'],phase=profile['phase'],**values,corpora=[c.meta for c in corpora],step=step,phase_tokens=step*32768*profile['batch'],tokens=initial_tokens+step*32768*profile['batch'],optimizer_steps_offset=80,parent_training_tokens=initial_tokens,parent_checkpoint_sha256=profile['parent_checkpoint_sha256'])
    torch.save(state,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt');seconds=time.monotonic()-tick
    row=dict(step=step,phase_tokens=state['phase_tokens'],tokens=state['tokens'],lm=lm.item(),balance=balance.item(),grad_norm=norm.item(),step_seconds=seconds,tokens_per_second=32768*profile['batch']/seconds,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,validation_pending=True,status='updates_complete_validation_pending' if step==2 else 'one_update_accepted_validation_pending')
    with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    if profile.get('migration_checkpoint'):assert digest(profile['migration_checkpoint'])==profile['migration_checkpoint_sha256'],'Private mmap writes altered old input checkpoint'


if __name__=='__main__':main()
