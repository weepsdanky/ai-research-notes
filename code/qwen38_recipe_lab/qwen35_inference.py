"""Read-only GDN/FA/SWA caches for the independent Qwen3.5 controls.

Explicit grouped dense GEMM, split-half partial RoPE and physical SWA key tails.
No QSA, fused sparse kernel, parameter changes or training-source modification.
"""
import math
import torch
from qwen35_controls import GDN35


def batch_rope(x,positions,dim,theta):
    frequency=theta**(-torch.arange(0,dim,2,device=x.device).float()/dim)
    angle=positions.float()[...,None]*frequency
    cos=torch.cat((angle.cos(),angle.cos()),-1)[:,:,None]
    sin=torch.cat((angle.sin(),angle.sin()),-1)[:,:,None]
    a,b=x[...,:dim//2],x[...,dim//2:dim]
    return torch.cat((x[...,:dim]*cos+torch.cat((-b,a),-1)*sin,x[...,dim:]),-1)


class ControlInference:
    def __init__(self,model,query_chunk=16,max_context=32768):
        assert query_chunk>0 and max_context>0
        self.model=model.eval();self.query_chunk=query_chunk;self.max_context=max_context;self.owner=object()
    def empty(self,batch=1):
        device=self.model.embedding.weight.device
        return dict(owner=self.owner,lengths=torch.zeros(batch,dtype=torch.long,device=device),layers=[{} for _ in self.model.layers])
    def fork(self,cache):
        assert cache['owner'] is self.owner
        # Every update below is out-of-place; immutable tensor storage may be shared.
        return dict(owner=self.owner,lengths=cache['lengths'],layers=[dict(layer) for layer in cache['layers']])
    def pack(self,caches):
        assert caches and all(c['owner'] is self.owner and len(c['lengths'])==1 for c in caches)
        result=self.empty(len(caches));result['lengths']=torch.cat([c['lengths'] for c in caches])
        for index,layer in enumerate(self.model.layers):
            states=[c['layers'][index] for c in caches]
            if isinstance(layer.mixer,GDN35):
                result['layers'][index]={key:torch.cat([s[key] for s in states],0) for key in ['conv','state']}
            else:
                first=states[0];length=max(s['positions'].shape[1] for s in states);b=len(caches)
                keys=first['k'].new_zeros(b,*first['k'].shape[1:3],length)
                values=first['v'].new_zeros(b,first['v'].shape[1],length,first['v'].shape[-1])
                positions=first['positions'].new_full((b,length),-1)
                for row,state in enumerate(states):
                    count=state['positions'].shape[1]
                    if count:
                        keys[row,:,:,length-count:]=state['k'][0]
                        values[row,:,length-count:]=state['v'][0]
                        positions[row,length-count:]=state['positions'][0]
                result['layers'][index]=dict(k=keys,v=values,positions=positions)
        return result
    def _attention(self,module,x,positions,state):
        c=module.c;b,t,_=x.shape;groups=c.heads//c.kv_heads
        q,gate=module.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
        q=batch_rope(module.qnorm(q),positions,c.rotary_dim,c.rope_theta).permute(0,2,1,3).reshape(b,c.kv_heads,groups,t,c.head_dim)
        k=batch_rope(module.knorm(module.k(x).reshape(b,t,c.kv_heads,c.head_dim)),positions,c.rotary_dim,c.rope_theta).permute(0,2,3,1)
        v=module.v(x).reshape(b,t,c.kv_heads,c.head_dim).permute(0,2,1,3)
        if state:
            k=torch.cat((state['k'],k),-1);v=torch.cat((state['v'],v),-2);key_positions=torch.cat((state['positions'],positions),-1)
        else:key_positions=positions
        pieces=[]
        for start in range(0,t,self.query_chunk):
            end=min(t,start+self.query_chunk);queries=q[:,:,:,start:end].reshape(b,c.kv_heads,-1,c.head_dim)
            delta=positions[:,start:end,None]-key_positions[:,None,:]
            valid=(key_positions[:,None,:]>=0)&(delta>=0)
            if module.window is not None:valid=valid&(delta<module.window)
            mask=valid.repeat(1,groups,1)[:,None]
            attention=((queries@k)/math.sqrt(c.head_dim)).masked_fill(~mask,-1e9).softmax(-1)@v
            pieces.append(attention.reshape(b,c.kv_heads,groups,end-start,c.head_dim))
        y=torch.cat(pieces,3).permute(0,3,1,2,4).reshape(b,t,c.heads,c.head_dim)*gate.sigmoid()
        if module.window is not None:
            keep=module.window-1
            k=k[...,-keep:] if keep else k[...,:0]
            v=v[...,-keep:,:] if keep else v[...,:0,:]
            key_positions=key_positions[:,-keep:] if keep else key_positions[:,:0]
        state.update(k=k,v=v,positions=key_positions)
        return module.out(y.flatten(-2))
    @torch.no_grad()
    def forward(self,ids,cache):
        assert cache['owner'] is self.owner and ids.ndim==2 and ids.shape[1]>0
        assert ids.shape[0]==len(cache['lengths']) and ids.device==self.model.embedding.weight.device
        assert bool((cache['lengths']+ids.shape[1]<=self.max_context).all()),'Context capacity exceeded'
        positions=cache['lengths'][:,None]+torch.arange(ids.shape[1],device=ids.device)[None]
        hidden=self.model.embedding(ids)
        for layer,state in zip(self.model.layers,cache['layers'],strict=True):
            normalized=layer.input_norm(hidden)
            mixed=layer.mixer(normalized,state) if isinstance(layer.mixer,GDN35) else self._attention(layer.mixer,normalized,positions,state)
            hidden=hidden+mixed;value,_=layer.moe(layer.post_norm(hidden));hidden=hidden+value
        cache['lengths']=cache['lengths']+ids.shape[1]
        return self.model.head(self.model.norm(hidden)[:,-1])
    def prefill(self,ids,chunk=128,cache=None):
        assert chunk>0 and ids.shape[1]>0
        cache=self.empty(ids.shape[0]) if cache is None else cache
        for start in range(0,ids.shape[1],chunk):logits=self.forward(ids[:,start:start+chunk],cache)
        return logits,cache
