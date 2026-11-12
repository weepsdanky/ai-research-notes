"""Formal20x32K updates with CPU substep checkpoints and unchanged math."""
import argparse,json,mmap,os,resource,signal,tempfile,time
from pathlib import Path
import torch
from qwen35_controls import ControlConfig
from qwen35_tiled_update import tiled_control
from qwen35_formal_long import parent_state
from qwen35_long_phase import exact
from qwen35_formal_contract import contract
from train_qwen35_controls import Corpus,digest
from staged_qwen35_update import StagedControlUpdate
from optim import RecipeOptimizer,ownership
from qwen35_quality_data import load,write_json


def migrate(profile_path,profile,out):
    m=profile['migration'];assert not out.exists()
    for key in ['legacy_profile','source_checkpoint','source_audit','replay_result']:
        assert digest(m[key])==m[key+'_sha256']
    proof=load(m['replay_result']);assert proof['status']=='passed_actual_32768_token_oracle_bitwise' and proof['oracle_sha256']==m['source_checkpoint_sha256']
    assert proof['model_optimizer_rng_bitwise_equal'] and proof['loss_balance_grad_norm_exact'] and proof['accepted_new_formal_updates']==0
    receipt=load(m['source_audit']);assert receipt['checkpoint_sha256']==m['source_checkpoint_sha256'] and receipt['step']==8 and receipt['status']=='passed_checkpoint_checks'
    previous=torch.load(m['source_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    assert previous['profile']==load(m['legacy_profile']) and previous['step']==m['credit_steps']==8 and previous['tokens']==3538944
    parent=parent_state(profile);assert previous['corpora']==parent['corpora']
    temporary=Path(tempfile.mkdtemp(prefix=out.name+'-migration-',dir=out.parent))
    identity=dict(profile=profile,profile_sha256=digest(profile_path),source_hashes=profile['source_hashes'])
    initial=dict(previous,**identity);initial.update({k:parent[k] for k in ['model','optimizer','rng','corpora']});initial.update(step=0,phase_tokens=0,tokens=parent['tokens'])
    torch.save(initial,temporary/'initial.pt');del initial,parent
    torch.save(dict(previous,**identity),temporary/'last.pt')
    saved=torch.load(temporary/'last.pt',map_location='cpu',weights_only=True,mmap=True)
    for key in ['model','optimizer','rng','corpora']:exact(saved[key],previous[key])
    assert saved['profile']==profile and saved['step']==8 and not out.exists()
    write_json(temporary/'migration.json',dict(status='exact_metadata_migration',credit_steps=8,accepted_new_updates=0,legacy_checkpoint_sha256=m['source_checkpoint_sha256'],migrated_checkpoint_sha256=digest(temporary/'last.pt'),replay_result_sha256=m['replay_result_sha256'],model_optimizer_rng_corpora_bitwise_preserved=True))
    temporary.rename(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--migration-only',action='store_true');a=p.parse_args()
    assert os.getppid()==int(os.environ['QWEN35_STAGED_FORMAL_GUARD_PID']) and 30<=a.seconds<=1140
    assert all(os.environ.get(k) is None for k in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
    requests=[];signal.signal(signal.SIGINT,lambda *_:requests.append('SIGINT'));signal.signal(signal.SIGTERM,lambda *_:requests.append('SIGTERM'))
    profile=load(a.profile);contract(profile,'formal_long');assert profile['training_execution']=='formal_staged_update' and profile['runtime_tiling']==dict(query_chunk=32,head_chunk=8)
    assert torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE
    torch.set_num_threads(1);torch.manual_seed(profile['seed']);out=Path(profile['out'])
    if a.migration_only:return migrate(a.profile,profile,out)
    cp=out/'last.pt';cp_sha=digest(cp);pending=out/'pending.pt'
    saved=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
    assert saved['profile']==profile and saved['profile_sha256']==digest(a.profile) and 8<=saved['step']<20
    model=tiled_control(ControlConfig(**profile['config']),profile['arm'],profile['seed'],32).train();model.load_state_dict(saved['model'],strict=True)
    corpora=[Corpus(row['directory'],row['meta_sha256'],32768) for row in profile['data']];assert [c.meta for c in corpora]==saved['corpora']
    rng=torch.Generator();rng.set_state(saved['rng']);index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
    x,y=corpora[index].batch('train',profile['batch'],32768,rng,'cpu')
    old=None
    if pending.exists():
        old=torch.load(pending,map_location='cpu',weights_only=True,mmap=True)
        assert old['checkpoint_sha256']==cp_sha and old['profile_sha256']==saved['profile_sha256'] and old['target_step']==saved['step']+1
        assert torch.equal(old['rng_after_draw'],rng.get_state())
    update=StagedControlUpdate(model,x,y,binding=cp_sha,chunk=8,balance_weight=profile['balance_weight'],state=None if old is None else old['update'])
    start=update.s['units'];tick=time.monotonic();prior_seconds=0 if old is None else old['active_seconds'];prior_peak=0 if old is None else old['peak_rss_bytes']
    while not update.done and not requests and time.monotonic()-tick<a.seconds:update.advance()
    peak=max(prior_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not update.done or requests:
        if update.s['units']>start or not pending.exists():
            torch.save(dict(checkpoint_sha256=cp_sha,profile_sha256=saved['profile_sha256'],target_step=saved['step']+1,rng_after_draw=rng.get_state(),update=update.snapshot(),active_seconds=prior_seconds+time.monotonic()-tick,peak_rss_bytes=peak),out/'pending.tmp');(out/'pending.tmp').replace(pending)
        print(json.dumps(dict(accepted_new_updates=0,start_units=start,units=update.s['units'],phase=update.s['phase'],ce_chunks=update.s['ce_done'],requested=requests)),flush=True)
        return
    opt=RecipeOptimizer(ownership(model),profile['lr'],weight_decay=profile['weight_decay'],ns_dtype=profile['ns_dtype']);opt.load_state_dict(saved['optimizer'])
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),profile['clip'],error_if_nonfinite=True,foreach=False);opt.step();step=saved['step']+1
    phase_tokens=step*32768*profile['batch'];total=saved['parent_training_tokens']+phase_tokens
    state=dict(saved,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),step=step,phase_tokens=phase_tokens,tokens=total)
    torch.save(state,out/'last.tmp');(out/'last.tmp').replace(cp)
    if pending.exists():pending.unlink()
    seconds=prior_seconds+time.monotonic()-tick
    row=dict(step=step,phase_tokens=phase_tokens,tokens=total,lm=update.s['lm'].item(),balance=update.s['balance'].item(),grad_norm=norm.item(),step_seconds=seconds,tokens_per_second=32768*profile['batch']/seconds,process_peak_rss_bytes=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),runtime_tiling=profile['runtime_tiling'],validation_pending=True,timing_scope='sum active stages plus final save; excludes reload, pending saves, audit and recovery')
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
