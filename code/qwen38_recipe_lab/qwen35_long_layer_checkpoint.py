"""CPU32K continuation of completed independent4K controls.

Keep the short-phase sources/checkpoints immutable. This bounded Mac phase uses
2x32768 tokens after80x4096 (the public400B:80B token ratio, not private scale or
schedule). Model, optimizer state and private data RNG carry across the boundary.
"""
import argparse,json,os,platform,resource,time
from pathlib import Path
import torch
from qwen35_controls import ControlConfig
from qwen35_layer_checkpoint import LayerCheckpointControl as Qwen35Control
from optim import RecipeOptimizer,ownership
from train_qwen35_controls import Corpus,balance_loss,evaluate,digest
from audit_qwen35_training import finite
from chunked_lm_loss import chunked_cross_entropy


def exact(a,b):
    if isinstance(a,torch.Tensor):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b,strict=True):exact(x,y)
    else:assert a==b


def load_parent(profile):
    path=profile['parent_checkpoint'];assert digest(path)==profile['parent_checkpoint_sha256']
    assert digest(profile['parent_audit'])==profile['parent_audit_sha256']
    receipt=json.loads(Path(profile['parent_audit']).read_text())
    assert receipt['status']=='passed_checkpoint_checks' and receipt['phase_complete']
    assert receipt['checkpoint_sha256']==profile['parent_checkpoint_sha256']
    parent=torch.load(path,map_location='cpu',weights_only=True);short=parent['profile']
    assert receipt['profile_sha256']==parent['profile_sha256']==digest(profile['parent_profile'])
    assert receipt['auditor_sha256']==digest('audit_qwen35_training.py') and receipt['source_hashes']==parent['source_hashes']
    assert parent['step']==short['steps']==80 and short['seq']==4096
    assert parent['tokens']==80*4096*short['batch']
    assert all(digest(p)==sha for p,sha in parent['source_hashes'].items())
    for key in ['arm','config','data','batch','seed','device','lr','weight_decay','ns_dtype','clip','balance_weight','mix_probability','schedule','precision','head_chunk']:
        assert profile[key]==short[key],f'Phase boundary changes {key}'
    assert profile['device']=='cpu' and profile['seq']==32768 and profile['steps']==2
    finite(parent['model']);finite(parent['optimizer'])
    return parent


@torch.no_grad()
def evaluate_both(model,corpora,profile):
    results={f'val32k/{key}':value for key,value in evaluate(model,corpora,profile,'cpu').items()}
    short=dict(profile,seq=4096)
    results.update({f'val4k/{key}':value for key,value in evaluate(model,corpora,short,'cpu').items()})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert os.getppid()==int(os.environ['QWEN35_CONTROL_GUARD_PID'])
    assert os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None
    profile=json.loads(a.profile.read_text());assert profile['phase']=='32k_continuation' and 1<=a.seconds<=1140
    assert profile['execution_variant']=='whole_layer_reentrant_checkpoint'
    assert {'qwen35_long_layer_checkpoint.py','qwen35_layer_checkpoint.py','run_qwen35_long_layer_guarded.py'}<=profile['source_hashes'].keys()
    assert profile['status']=='resource_probe_candidate' and profile['scope']=='Mac public-family control, not private Table1'
    assert {'qwen35_long_phase.py','audit_qwen35_long_phase.py','run_qwen35_long_guarded.py','train_qwen35_controls.py','qwen35_controls.py','optim.py','model.py','polar_coefficients.json','chunked_lm_loss.py','audit_qwen35_training.py'}<=profile['source_hashes'].keys()
    assert all(digest(path)==sha for path,sha in profile['source_hashes'].items())
    torch.set_num_threads(1);torch.manual_seed(profile['seed']);parent=load_parent(profile)
    config=ControlConfig(**profile['config']);corpora=[Corpus(row['directory'],row['meta_sha256'],profile['seq']) for row in profile['data']]
    assert [c.meta for c in corpora]==parent['corpora']
    out=Path(profile['out']);checkpoint=out/'last.pt';saved=None
    if a.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['profile_sha256']==digest(a.profile) and saved['source_hashes']==profile['source_hashes']
        assert saved['parent_checkpoint_sha256']==profile['parent_checkpoint_sha256'] and 0<=saved['step']<2
    else:out.mkdir(parents=True,exist_ok=False)
    model=Qwen35Control(config,profile['arm'],seed=profile['seed'])
    optimizer=RecipeOptimizer(ownership(model),profile['lr'],weight_decay=profile['weight_decay'],ns_dtype=profile['ns_dtype'])
    initial=parent if saved is None else saved
    model.load_state_dict(initial['model'],strict=True);optimizer.load_state_dict(initial['optimizer'])
    rng=torch.Generator();rng.set_state(initial['rng']);step=0 if saved is None else saved['step']
    if saved is None:
        exact(model.state_dict(),parent['model']);exact(optimizer.state_dict(),parent['optimizer']);exact(rng.get_state(),parent['rng'])
    parent_tokens=parent['tokens'];parent_steps=parent['step'];del saved,initial,parent
    def log(row):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def save():
        state=dict(profile=profile,profile_sha256=digest(a.profile),source_hashes=profile['source_hashes'],config=profile['config'],arm=profile['arm'],phase=profile['phase'],model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),corpora=[c.meta for c in corpora],step=step,phase_tokens=step*profile['seq']*profile['batch'],tokens=parent_tokens+step*profile['seq']*profile['batch'],optimizer_steps_offset=parent_steps,parent_training_tokens=parent_tokens,parent_checkpoint_sha256=profile['parent_checkpoint_sha256'])
        torch.save(state,out/'last.tmp');(out/'last.tmp').replace(checkpoint)
    if not a.resume:
        save();os.link(checkpoint,out/'initial.pt')  # Immutable boundary proof before any forward/update.
        log(dict(step=0,phase_tokens=0,tokens=parent_tokens,**evaluate_both(model,corpora,profile)))
    began=time.monotonic();model.train()
    for next_step in range(step+1,profile['steps']+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        index=int(torch.rand((),generator=rng).item()<profile['mix_probability'])
        x,y=corpora[index].batch('train',profile['batch'],profile['seq'],rng,'cpu')
        hidden,routers=model.hidden(x)
        lm=chunked_cross_entropy(hidden,model.head.weight,y,profile['head_chunk'])
        balance=balance_loss(routers,config.experts,config.top_k);loss=lm+profile['balance_weight']*balance
        assert torch.isfinite(loss).item();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),profile['clip'],error_if_nonfinite=True,foreach=False);optimizer.step()
        step=next_step;save();seconds=time.monotonic()-tick
        stop=step==profile['steps'] or time.monotonic()-began>=a.seconds
        row=dict(step=step,phase_tokens=step*profile['seq']*profile['batch'],tokens=parent_tokens+step*profile['seq']*profile['batch'],lm=lm.item(),balance=balance.item(),grad_norm=norm.item(),step_seconds=seconds,tokens_per_second=profile['seq']*profile['batch']/seconds,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024))
        if stop:row.update(evaluate_both(model,corpora,profile))
        log(row)
        if stop:log(dict(step=step,status='complete' if step==profile['steps'] else 'time_budget',checkpoint=str(checkpoint)));break


if __name__=='__main__':main()
