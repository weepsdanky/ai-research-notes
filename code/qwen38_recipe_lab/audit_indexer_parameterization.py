"""Check learned-norm reparameterization, without claiming training equivalence.

For s=d**(-1/4), scaling both affine RMSNorm weights by s makes the bare
head-sum score equal the HF score divided by sqrt(d). Optimizer trajectories
are not equivalent without further transformations; no checkpoint is changed.
"""
import hashlib
import json
from pathlib import Path
import torch
from model import Config,Indexer,kl_loss


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    torch.set_num_threads(1);torch.manual_seed(112)
    path=Path('runs/qwen-sparse/last.pt');checkpoint_sha=digest(path)
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    c=Config(**saved['config']);assert c.score_mode=='hf'
    prefixes={k.removesuffix('qnorm.weight') for k in saved['model'] if k.endswith('.indexer.qnorm.weight')}
    assert len(prefixes)==1
    prefix=next(iter(prefixes));state={k.removeprefix(prefix):v for k,v in saved['model'].items() if k.startswith(prefix)}
    hf=Indexer(c);hf.load_state_dict(state);bare=Indexer(Config(**{**saved['config'],'score_mode':'paper'}));bare.load_state_dict(state)
    factor=c.index_dim**(-.25)
    with torch.no_grad():
        bare.qnorm.weight.add_(1).mul_(factor).sub_(1);bare.knorm.weight.add_(1).mul_(factor).sub_(1)
    del state,saved
    results=[]
    for length,offset in [(128,0.),(128,.5),(512,0.)]:
        x=torch.randn(2,length,c.width)+offset
        a,valid,selected,tokens=hf(x);b,valid_b,selected_b,tokens_b=bare(x)
        torch.testing.assert_close(a,b,atol=5e-5,rtol=2e-4)
        assert torch.equal(valid,valid_b) and torch.equal(selected,selected_b) and torch.equal(tokens,tokens_b)
        teacher=torch.rand_like(a);eligible=valid[None].expand_as(a)
        loss_a=kl_loss(teacher,a,eligible);loss_b=kl_loss(teacher,b,eligible)
        grad_a=torch.autograd.grad(loss_a,list(hf.parameters()));grad_b=torch.autograd.grad(loss_b,list(bare.parameters()))
        errors={}
        for (name,_),ga,gb in zip(hf.named_parameters(),grad_a,grad_b):
            expected=gb*factor if name in ('qnorm.weight','knorm.weight') else gb
            torch.testing.assert_close(ga,expected,atol=3e-6,rtol=3e-4)
            errors[name]=float((ga-expected).abs().max())
        results.append(dict(length=length,input_offset=offset,valid_score_max_abs_error=float((a-b)[eligible].abs().max().detach()),
            kl_abs_error=float((loss_a-loss_b).abs().detach()),selected_blocks_and_tokens_exact=True,gradient_max_abs_errors=errors))
    assert digest(path)==checkpoint_sha
    report=dict(checkpoint_sha256=checkpoint_sha,model_source_sha256=digest('model.py'),probe_source_sha256=digest(__file__),indexer_prefix=prefix,
        index_dim=c.index_dim,norm_weight_factor=factor,results=results,
        scope='Actual trained indexer weights, synthetic hidden inputs/teacher; CPU FP32. No backbone, main weights or optimizer state changed.',
        conclusions=['Forward score families can be mapped through learned affine norm weights.',
            'Norm gradients have inverse scale; equal numeric optimizer settings and clipping do not imply equal training.',
            'This does not repair or pass the failed cold-start paper-profile training acceptance.'])
    Path('runs/indexer-parameterization-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
