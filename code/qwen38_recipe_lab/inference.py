"""Incremental eager KV reference for evaluation, not a fused sparse kernel.

Training and its hash-locked implementation remain in model.py. Prefill calls
that exact attention path; decode caches rotated core KV and pooled index keys.
All predictions are checked against full-prefix evaluation in test_inference.py.
"""
import math
import torch
from model import Attention, GatedResidual, rope


class Inference:
    def __init__(self,model,stage,prefill_chunk=0,grouped_gqa=False):
        if stage not in ('pretrain','sparse'): raise ValueError('Inference stage must be dense or sparse')
        self.model,self.stage=model,stage
        self.prefill_chunk=prefill_chunk
        self.grouped_gqa=grouped_gqa

    def attention(self,m,x,cache):
        c=m.c; b,t,_=x.shape
        if not cache and (not self.grouped_gqa or self.stage!='sparse'):
            output=m(x,self.stage)[0]
            positions=torch.arange(t,device=x.device)
            cache['k']=rope(m.knorm(m.k(x).reshape(b,t,c.kv_heads,c.head_dim)),positions,c.rotary_dim,c.rope_theta)
            cache['v']=m.v(x).reshape(b,t,c.kv_heads,c.head_dim)
            if self.stage=='sparse':
                raw=m.indexer.qk(x)[...,-c.index_dim:]
                n=t//c.block*c.block
                cache['tail']=raw[:,n:]
                pooled=raw[:,:n].reshape(b,n//c.block,c.block,c.index_dim).float().mean(2)
                cache['pooled']=rope(m.indexer.knorm(pooled)[:,:,None],torch.arange(n//c.block,device=x.device)*c.block,c.rotary_dim,c.rope_theta)[:,:,0]
            return output
        if not cache:
            cache.update(k=x.new_empty((b,0,c.kv_heads,c.head_dim)),v=x.new_empty((b,0,c.kv_heads,c.head_dim)),
                         tail=x.new_empty((b,0,c.index_dim)),pooled=x.new_empty((b,0,c.index_dim),dtype=torch.float32))
        pos=cache['k'].shape[1]; position=torch.arange(pos,pos+t,device=x.device)
        q,gate=m.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=rope(m.qnorm(q),position,c.rotary_dim,c.rope_theta).transpose(1,2)
        k=rope(m.knorm(m.k(x).reshape(b,t,c.kv_heads,c.head_dim)),position,c.rotary_dim,c.rope_theta)
        cache['k']=torch.cat((cache['k'],k),1)
        cache['v']=torch.cat((cache['v'],m.v(x).reshape(b,t,c.kv_heads,c.head_dim)),1)
        if self.stage=='sparse':
            index=m.indexer
            iq,raw=index.qk(x).split((c.index_heads*c.index_dim,c.index_dim),-1)
            iq=rope(index.qnorm(iq.reshape(b,t,c.index_heads,c.index_dim)),position,c.rotary_dim,c.rope_theta)
            joined=torch.cat((cache['tail'],raw),1);n=joined.shape[1]//c.block
            if n:
                pooled=index.knorm(joined[:,:n*c.block].reshape(b,n,c.block,c.index_dim).float().mean(2))[:,:,None]
                starts=(torch.arange(n,device=x.device)+cache['pooled'].shape[1])*c.block
                pooled=rope(pooled,starts,c.rotary_dim,c.rope_theta)[:,:,0]
                cache['pooled']=torch.cat((cache['pooled'],pooled),1)
            cache['tail']=joined[:,n*c.block:]
            scores=torch.einsum('bthd,bsd->bths',iq,cache['pooled']).relu().sum(2)
            if c.score_mode=='hf': scores=scores/math.sqrt(c.index_dim)
            valid=(torch.arange(scores.shape[-1],device=x.device)*c.block+c.block-1)[None]<=position[:,None]
            scores=scores.masked_fill(~valid,-1e9)
            count=min(c.budget//c.block,scores.shape[-1])
            selected=scores.argsort(dim=-1,descending=True,stable=True)[...,:count]
            selected=selected.masked_fill(~valid[None].expand(b,-1,-1).gather(-1,selected),-1)
            tokens=(selected[...,None]*c.block+torch.arange(c.block,device=x.device)).flatten(-2)
            tokens=tokens.masked_fill(~(selected>=0).repeat_interleave(c.block,-1),-1)
            tail=((position+1)//c.block*c.block)[:,None]+torch.arange(c.block-1,device=x.device)
            tail=tail.masked_fill(tail>position[:,None],-1)[None].expand(b,-1,-1)
            tokens=torch.cat((tokens,tail),-1)
        else:
            keys=cache['k'].repeat_interleave(c.heads//c.kv_heads,2).transpose(1,2)
            values=cache['v'].repeat_interleave(c.heads//c.kv_heads,2).transpose(1,2)
            weights=(q@keys.transpose(-1,-2)/math.sqrt(c.head_dim)).masked_fill(torch.arange(pos+t,device=x.device)[None]>position[:,None],-1e9).softmax(-1)
            y=(weights@values).transpose(1,2)*gate.sigmoid()
            return m.out(y.flatten(-2))
        batch=torch.arange(b,device=x.device)[:,None,None]
        if self.grouped_gqa:
            # Contract query groups against each shared KV head. Avoid copying
            # gathered K/V once per query head or materializing q*k products.
            keys=cache['k'][batch,tokens.clamp_min(0)].permute(0,1,3,4,2)
            values=cache['v'][batch,tokens.clamp_min(0)].permute(0,1,3,2,4)
            queries=q.transpose(1,2).reshape(b,t,c.kv_heads,c.heads//c.kv_heads,c.head_dim)
            scores=(queries@keys)/math.sqrt(c.head_dim)
            weights=scores.masked_fill(tokens[:,:,None,None]<0,-1e9).softmax(-1)
            y=(weights@values).reshape(b,t,c.heads,c.head_dim)*gate.sigmoid()
            return m.out(y.flatten(-2))
        keys=cache['k'][batch,tokens.clamp_min(0)].permute(0,3,1,2,4).repeat_interleave(c.heads//c.kv_heads,1)
        values=cache['v'][batch,tokens.clamp_min(0)].permute(0,3,1,2,4).repeat_interleave(c.heads//c.kv_heads,1)
        weights=((q[...,None,:]*keys).sum(-1)/math.sqrt(c.head_dim)).masked_fill(tokens[:,None]<0,-1e9).softmax(-1)
        y=(weights[...,None]*values).sum(-2).transpose(1,2)*gate.sigmoid()
        return m.out(y.flatten(-2))

    def hidden(self,ids,cache):
        model=self.model
        residual=model.embedding(ids).repeat(1,1,model.config.branches)
        for i,layer in enumerate(model.layers):
            local=cache.setdefault(str(i),{})
            if layer.ple is not None: residual=residual+layer.ple(residual,ids,local.setdefault('ple',{}))
            x,write=layer.mix_residual(residual)
            slot=local.setdefault('mix',{})
            y=self.attention(layer.mix,x,slot) if isinstance(layer.mix,Attention) else layer.mix(x,slot)
            residual=GatedResidual.add(residual,y,write)
            x,write=layer.moe_residual(residual)
            residual=GatedResidual.add(residual,layer.moe(x)[0],write)
        return residual

    @torch.no_grad()
    def __call__(self,ids,cache):
        if ids.shape[1]<1:raise ValueError('Empty token input')
        chunk=self.prefill_chunk if not cache else 0
        if chunk and ids.shape[1]>chunk:
            for part in ids.split(chunk,dim=1):residual=self.hidden(part,cache)
        else:residual=self.hidden(ids,cache)
        return self.model.head(self.model.readout(residual[:,-1:]))[:,0]
