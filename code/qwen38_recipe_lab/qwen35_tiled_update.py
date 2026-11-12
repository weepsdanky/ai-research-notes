"""One32K update with explicitly recorded query/head tiling, unchanged math.

Fresh runs start from the audited4K parent. Existing exploratory checkpoints and
frozen numerical engines remain untouched; this is a distinct execution profile.
"""
import argparse,json,mmap,os,resource,time
from dataclasses import replace
from pathlib import Path
import torch
from qwen35_controls import ControlConfig
from qwen35_layer_checkpoint import LayerCheckpointControl
from qwen35_streamed_update import parent_state
from qwen35_long_phase import exact
from train_qwen35_controls import Corpus,balance_loss,digest
from streamed_lm_backward import streamed_lm_backward
from optim import RecipeOptimizer,ownership


def tiled_control(config,arm,seed,query_chunk):
    assert query_chunk>0
    return LayerCheckpointControl(replace(config,query_chunk=query_chunk),arm,seed=seed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert os.getppid()==int(os.environ['QWEN35_CONTROL_GUARD_PID']) and 1<=a.seconds<=1140
    profile=json.loads(Path(a.profile).read_text());assert profile['training_execution']=='tiled_streamed_one_update_separate_validation'
    assert profile['phase']=='32k_continuation' and profile['seq']==32768 and profile['steps']==2 and profile['device']=='cpu'
    runtime=profile['runtime_tiling'];assert runtime==dict(query_chunk=32,head_chunk=8)
    assert {'qwen35_tiled_update.py','run_qwen35_tiled_guarded.py','streamed_lm_backward.py','qwen35_streamed_update.py'}<=profile['source_hashes'].keys()
    assert all(digest(p)==sha for p,sha in profile['source_hashes'].items())
    assert all(os.environ.get(k) is None for k in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_ENABLE_MPS_FALLBACK'])
    assert torch.serialization.get_default_mmap_options()==mmap.MAP_PRIVATE
    torch.set_num_threads(1);torch.manual_seed(profile['seed']);parent=parent_state(profile);out=Path(profile['out'])
    if not a.resume:
        from datetime import datetime
        admission=profile['resource_readmission'];samples=json.loads(Path(admission['samples_file']).read_text())
        assert digest(admission['samples_file'])==admission['samples_sha256'] and len(samples)==12 and all(s['pressure']==1 and s['free']>=55 for s in samples)
        assert 0<=(datetime.now().astimezone()-datetime.fromisoformat(samples[-1]['at'])).total_seconds()<=120
        out.mkdir(parents=True,exist_ok=False)
    corpora=[Corpus(row['directory'],row['meta_sha256'],32768) for row in profile['data']];assert [c.meta for c in corpora]==parent['corpora']
    parent_tokens=parent['tokens']
    def wrapped(values,step):
        return dict(profile=profile,profile_sha256=digest(a.profile),source_hashes=profile['source_hashes'],config=profile['config'],runtime_tiling=runtime,arm=profile['arm'],phase=profile['phase'],model=values['model'],optimizer=values['optimizer'],rng=values['rng'],corpora=[c.meta for c in corpora],step=step,phase_tokens=step*32768*profile['batch'],tokens=parent_tokens+step*32768*profile['batch'],optimizer_steps_offset=80,parent_training_tokens=parent_tokens,parent_checkpoint_sha256=profile['parent_checkpoint_sha256'])
    if a.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True,mmap=True);assert saved['profile_sha256']==digest(a.profile) and saved['source_hashes']==profile['source_hashes'] and saved['runtime_tiling']==runtime
        step=saved['step']
    else:
        assert not profile.get('migration_checkpoint'),'This variant starts from the original4K parent'
        saved=parent;step=0;torch.save(wrapped(parent,0),out/'initial.pt');os.link(out/'initial.pt',out/'last.pt')
    assert 0<=step<2
    model=tiled_control(ControlConfig(**profile['config']),profile['arm'],profile['seed'],runtime['query_chunk']).train();model.load_state_dict(saved['model'],strict=True)
    assert model.head.weight.data_ptr()!=model.embedding.weight.data_ptr()
    optimizer=RecipeOptimizer(ownership(model),profile['lr'],weight_decay=profile['weight_decay'],ns_dtype=profile['ns_dtype']);optimizer.load_state_dict(saved['optimizer'])
    rng=torch.Generator();rng.set_state(saved['rng']);exact(model.state_dict(),saved['model']);exact(optimizer.state_dict(),saved['optimizer']);exact(rng.get_state(),saved['rng']);del parent,saved
    optimizer.zero_grad(set_to_none=True);tick=time.monotonic();index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
    x,y=corpora[index].batch('train',profile['batch'],32768,rng,'cpu');hidden,routers=model.hidden(x);balance=balance_loss(routers,model.config.experts,model.config.top_k);assert torch.isfinite(balance).item()
    lm,adjoint=streamed_lm_backward(hidden,model.head.weight,y,runtime['head_chunk']);torch.autograd.backward((hidden,balance),(adjoint,balance.new_tensor(profile['balance_weight'])))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),profile['clip'],error_if_nonfinite=True,foreach=False);optimizer.step();step+=1
    state=wrapped(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state()),step);torch.save(state,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt');seconds=time.monotonic()-tick
    row=dict(step=step,phase_tokens=state['phase_tokens'],tokens=state['tokens'],lm=lm.item(),balance=balance.item(),grad_norm=norm.item(),step_seconds=seconds,tokens_per_second=32768*profile['batch']/seconds,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,runtime_tiling=runtime,validation_pending=True)
    with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
