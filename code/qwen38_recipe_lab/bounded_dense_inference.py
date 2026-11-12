"""Dense GQA with explicit grouped GEMM, avoiding broadcast-matmul KV clones. CPU reference, not a fused sparse kernel."""
import math
import torch
from model import rope
from prefill_inference import PrefillInference


class BoundedDenseInference(PrefillInference):
    def __init__(self,model,stage='pretrain',prefill_chunk=128,grouped_gqa=False):
        assert stage=='pretrain' and not grouped_gqa
        super().__init__(model,stage,prefill_chunk,grouped_gqa)

    def attention(self,m,x,cache):
        c=m.c;b,t,_=x.shape;pos=cache['k'].shape[1] if cache else 0
        position=torch.arange(pos,pos+t,device=x.device)
        q,gate=m.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=rope(m.qnorm(q),position,c.rotary_dim,c.rope_theta).permute(0,2,1,3).reshape(b,c.kv_heads,c.heads//c.kv_heads,t,c.head_dim)
        k=rope(m.knorm(m.k(x).reshape(b,t,c.kv_heads,c.head_dim)),position,c.rotary_dim,c.rope_theta)
        v=m.v(x).reshape(b,t,c.kv_heads,c.head_dim)
        cache['k']=torch.cat((cache['k'],k),1) if cache else k
        cache['v']=torch.cat((cache['v'],v),1) if 'v' in cache else v
        keys=cache['k'].permute(0,2,3,1)
        values=cache['v'].permute(0,2,1,3)
        queries=q.reshape(b,c.kv_heads,c.heads//c.kv_heads*t,c.head_dim)
        scores=((queries@keys)/math.sqrt(c.head_dim)).reshape(b,c.kv_heads,c.heads//c.kv_heads,t,pos+t)
        valid=torch.arange(pos+t,device=x.device)[None]<=position[:,None]
        weights=scores.masked_fill(~valid,-1e9).softmax(-1)
        y=(weights.reshape(b,c.kv_heads,c.heads//c.kv_heads*t,pos+t)@values).reshape(b,c.kv_heads,c.heads//c.kv_heads,t,c.head_dim).permute(0,3,1,2,4).reshape(b,t,c.heads,c.head_dim)*gate.sigmoid()
        return m.out(y.flatten(-2))
