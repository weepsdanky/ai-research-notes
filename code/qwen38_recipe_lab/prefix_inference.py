"""Reuse exact token prefixes in no-grad evaluation; no fused-kernel claim."""
import torch


def common_prefix(rows, chunk):
    """Align reuse to existing prefill chunk boundaries, preserving scan order."""
    if not rows:return []
    prefix=list(rows[0])
    for row in rows[1:]:
        limit=min(len(prefix),len(row));i=0
        while i<limit and prefix[i]==row[i]:i+=1
        del prefix[i:]
        if len(prefix)<chunk:return []
    return prefix[:len(prefix)//chunk*chunk]


def clone_batch(tree, batch):
    if isinstance(tree,dict):return {k:clone_batch(v,batch) for k,v in tree.items()}
    if not isinstance(tree,torch.Tensor) or tree.ndim==0 or tree.shape[0]!=1:
        raise ValueError('Prefix cache must contain batch-one tensors')
    return tree.expand(batch,*tree.shape[1:]).clone()


class PrefixInference:
    @torch.no_grad()
    def __init__(self,runner,prefix):
        if not prefix or not runner.prefill_chunk or len(prefix)%runner.prefill_chunk:raise ValueError('Nonempty chunk-aligned prefix required')
        self.runner=runner;self.prefix=torch.tensor(prefix,device=next(runner.model.parameters()).device)
        self.seed={};self.last=runner(self.prefix[None],self.seed)

    @torch.no_grad()
    def __call__(self,ids,cache):
        if cache:return self.runner(ids,cache)
        n=len(self.prefix)
        if ids.shape[1]<n or not torch.equal(ids[:,:n],self.prefix[None].expand(ids.shape[0],-1)):
            raise ValueError('Input differs from the cached token prefix')
        cache.update(clone_batch(self.seed,ids.shape[0]))
        if ids.shape[1]==n:return self.last.expand(ids.shape[0],-1).clone()
        for part in ids[:,n:].split(self.runner.prefill_chunk,dim=1):last=self.runner(part,cache)
        return last
