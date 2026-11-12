"""Check useful indexer learning against an exact uniform-student KL baseline.

Skips the large vocabulary head: attention teachers and indexer inputs are still
obtained from every actual trained backbone layer. CPU by default to avoid
contending with a live MPS training process.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from model import Config,Model,Indexer
from train import Corpus


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return True


def distillation_integrity(saved,initial):
    assert saved['args']['stage']=='distill' and saved['step']==saved['args']['steps'], 'Complete distillation checkpoint required'
    assert initial['args']['stage']=='pretrain' and initial['step']==initial['args']['steps'], 'Complete pretraining initializer required'
    actual={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('model.py','optim.py','polar_coefficients.json','train.py')}
    assert saved['source_hashes']==initial['source_hashes']==actual, 'Training source mismatch'
    assert saved['model'].keys()==initial['model'].keys(), 'Model tensor keys differ'
    changed=[name for name,value in saved['model'].items() if not torch.equal(value,initial['model'][name])]
    assert all('.indexer.' in name for name in changed), 'Backbone changed during distillation'
    assert finite_tree(saved['model']) and finite_tree(saved['optimizers']), 'Nonfinite checkpoint tensor'
    return dict(all_non_indexer_tensors_bit_exact_to_init=True,changed_indexer_tensors=changed,
                all_model_and_optimizer_tensors_finite=True,training_source_hashes=actual)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--init',help='Also verify frozen backbone, finite state and exact training source hashes against the phase initializer'); p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--batches',type=int,default=8); p.add_argument('--out',required=True)
    a=p.parse_args(); torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    integrity=distillation_integrity(saved,torch.load(a.init,weights_only=True,map_location='cpu',mmap=True)) if a.init else None
    if integrity is not None:integrity.update(initial_checkpoint=str(Path(a.init).resolve()),initial_checkpoint_sha256=digest(a.init))
    model=Model(Config(**saved['config'])).eval(); model.load_state_dict(saved['model']); model.stage('distill')
    indexers=[m for m in model.modules() if isinstance(m,Indexer)]
    original=[m.qk.weight.detach().clone() for m in indexers]
    captured=[]; handles=[m.register_forward_hook(lambda m,i,o:captured.append(o)) for m in indexers]
    def body(ids):
        state=model.embedding(ids).repeat(1,1,model.config.branches); kl=state.new_zeros(())
        for layer in model.layers:
            state,local,_=layer(state,ids,'distill'); kl=kl+local
        return kl/len(indexers)
    report={}
    for directory in a.data:
        corpus=Corpus(directory,128); rng=torch.Generator().manual_seed(1234)
        values=[]; nonuniform=[]; score_nonzero=[]; uniform=[]
        for _ in range(a.batches):
            ids,_=corpus.batch('val',1,128,rng,'cpu'); captured.clear()
            actual=body(ids); model.zero_grad(); actual.backward()
            grad=sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None)**.5
            assert all('.indexer.' in name for name,p in model.named_parameters() if p.grad is not None)
            for scores,valid,_,_ in captured:
                mask=valid[None].expand_as(scores); count=mask.sum(-1)
                mean=scores.masked_fill(~mask,0).sum(-1)/count.clamp_min(1)
                variance=((scores-mean[...,None]).square()*mask).sum(-1)/count.clamp_min(1)
                usable=count>=2
                nonuniform.append(((variance>1e-8)&usable).sum().item()/usable.sum().item())
                score_nonzero.append((scores[mask]!=0).float().mean().item())
            values.append((actual.item(),grad))
            with torch.no_grad():
                for m in indexers: m.qk.weight.zero_()
                uniform.append(body(ids).item())
                for m,weight in zip(indexers,original,strict=True): m.qk.weight.copy_(weight)
        actual=sum(v[0] for v in values)/len(values); flat=sum(uniform)/len(uniform)
        nonuniform_fraction=sum(nonuniform)/len(nonuniform)
        report[directory]=dict(kl=actual,uniform_student_kl=flat,kl_better_than_uniform=actual<flat,
            nonuniform_query_fraction=nonuniform_fraction,nonzero_score_fraction=sum(score_nonzero)/len(score_nonzero),
            mean_indexer_grad_norm=sum(v[1] for v in values)/len(values),
            nondegenerate=actual<flat and nonuniform_fraction>=.5 and all(v[1]>0 for v in values))
    for h in handles: h.remove()
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=digest(a.checkpoint),
                auditor_sha256=digest(__file__),integrity=integrity,step=saved['step'],profile=saved['config']['score_mode'],
                query_criterion='count>=2; score variance>1e-8; at least 50% nonuniform; KL better than uniform; positive gradient every batch',corpora=report)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
