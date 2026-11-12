"""Public feature replacement, 3D packing and multimodal text forward.

Uses already-normalized vision patches; raw image/video processor and production
multimodal/MTP training schedules remain separate. No text checkpoint migration.
"""
import itertools,math
import torch
from torch import nn
from model import Attention,GatedResidual
from vision import text_mrope


def vision_positions(start,grid,spatial_merge=2,temporal_merge=1,interval=1.,device=None):
    t,h,w=[int(v) for v in grid]
    if min(t,h,w)<1 or t%temporal_merge or h%spatial_merge or w%spatial_merge:raise ValueError('Invalid vision grid')
    axes=((torch.arange(t//temporal_merge,device=device)*interval).long()+start,
          torch.arange(h//spatial_merge,device=device)+start,torch.arange(w//spatial_merge,device=device)+start)
    return torch.stack(torch.meshgrid(*axes,indexing='ij')).reshape(3,-1)


def multimodal_positions(ids,types,image_grid=None,video_grid=None,mask=None,merge=2):
    if ids.shape!=types.shape:raise ValueError('Token type shape mismatch')
    images=[] if image_grid is None else image_grid.tolist()
    videos=[] if video_grid is None else [[1,h,w] for t,h,w in video_grid.tolist() for _ in range(t)]
    grids={1:iter(images),2:iter(videos)};result=torch.zeros(3,*ids.shape,dtype=ids.dtype,device=ids.device);deltas=[]
    for batch in range(ids.shape[0]):
        valid=torch.ones(ids.shape[1],device=ids.device,dtype=torch.bool) if mask is None else mask[batch].bool()
        row=types[batch,valid].tolist();position=0;parts=[]
        for kind,group in itertools.groupby(row):
            length=len(list(group))
            if kind==0:
                parts.append(torch.arange(length,device=ids.device)[None].expand(3,-1)+position);position+=length
            elif kind in grids:
                try:grid=next(grids[kind])
                except StopIteration:raise ValueError('Missing modality grid') from None
                local=vision_positions(position,grid,merge,device=ids.device)
                if local.shape[1]!=length:raise ValueError('Vision run length does not match merged grid')
                parts.append(local);position+=max(grid[1:])//merge
            else:raise ValueError('Unknown modality type')
        if not parts:raise ValueError('Empty sequence')
        joined=torch.cat(parts,1);result[:,batch,valid]=joined
        deltas.append(int(joined.max())+1-len(row))
    return result,torch.tensor(deltas,device=ids.device,dtype=ids.dtype)[:,None]


def inject_features(ids,embedding,image=None,video=None,image_id=248056,video_id=248057):
    result=embedding(ids)
    for token,features in ((image_id,image),(video_id,video)):
        if features is None:continue
        selected=(ids==token)[...,None].expand_as(result)
        if features.ndim!=2 or features.shape[-1]!=result.shape[-1] or features.numel()!=int(selected.sum()):
            raise ValueError('Visual features and placeholder tokens do not match')
        result=result.masked_scatter(selected,features.to(device=result.device,dtype=result.dtype))
    return result


def rotate(x,cos,sin):
    dim=cos.shape[-1];a,b=x[...,:dim].chunk(2,-1)
    changed=x[...,:dim]*cos[:,:,None]+torch.cat((-b,a),-1)*sin[:,:,None]
    return torch.cat((changed,x[...,dim:]),-1)


def attention(m,x,positions,stage):
    c=m.c;b,t,_=x.shape;cos,sin=text_mrope(positions,c.rotary_dim,
        sections=(c.rotary_dim//2//3+(c.rotary_dim//2%3>0),c.rotary_dim//2//3+(c.rotary_dim//2%3>1),c.rotary_dim//2//3),theta=c.rope_theta,dtype=x.dtype)
    q,gate=m.q_gate(x).reshape(b,t,c.heads,2*c.head_dim).chunk(2,-1)
    q=rotate(m.qnorm(q),cos,sin).transpose(1,2)
    k=rotate(m.knorm(m.k(x).reshape(b,t,c.kv_heads,c.head_dim)),cos,sin).repeat_interleave(c.heads//c.kv_heads,2)
    v=m.v(x).reshape(b,t,c.kv_heads,c.head_dim).repeat_interleave(c.heads//c.kv_heads,2)
    pos=torch.arange(t,device=x.device)
    if stage=='sparse':
        index=m.indexer;blocks=t//c.block
        iq,raw=index.qk(x.detach()).split((c.index_heads*c.index_dim,c.index_dim),-1)
        iq=rotate(index.qnorm(iq.reshape(b,t,c.index_heads,c.index_dim)),cos,sin)
        pooled=raw[:,:blocks*c.block].reshape(b,blocks,c.block,c.index_dim).float().mean(2)
        ik=rotate(index.knorm(pooled)[:,:,None],cos[:,::c.block][:,:blocks],sin[:,::c.block][:,:blocks])[:,:,0]
        scores=torch.einsum('bthd,bsd->bths',iq,ik).relu().sum(2)
        if c.score_mode=='hf':scores=scores/math.sqrt(c.index_dim)
        valid=(torch.arange(blocks,device=x.device)*c.block+c.block-1)[None]<=pos[:,None]
        chosen=scores.masked_fill(~valid,-1e9).argsort(dim=-1,descending=True,stable=True)[...,:min(c.budget//c.block,blocks)]
        chosen=chosen.masked_fill(~valid[None].expand(b,-1,-1).gather(-1,chosen),-1)
        tokens=(chosen[...,None]*c.block+torch.arange(c.block,device=x.device)).flatten(-2)
        tokens=tokens.masked_fill(~(chosen>=0).repeat_interleave(c.block,-1),-1)
        tail=((pos+1)//c.block*c.block)[:,None]+torch.arange(c.block-1,device=x.device)
        tail=tail.masked_fill(tail>pos[:,None],-1)[None].expand(b,-1,-1)
        tokens=torch.cat((tokens,tail),-1)
        batch=torch.arange(b,device=x.device)[:,None,None]
        keys=k[batch,tokens.clamp_min(0)].permute(0,3,1,2,4)
        values=v[batch,tokens.clamp_min(0)].permute(0,3,1,2,4)
        weights=((q[...,None,:]*keys).sum(-1)/math.sqrt(c.head_dim)).masked_fill(tokens[:,None]<0,-1e9).softmax(-1)
        y=(weights[...,None]*values).sum(-2)
    elif stage=='pretrain':
        weights=(q@k.transpose(1,2).transpose(-1,-2)/math.sqrt(c.head_dim)).masked_fill(pos[None]>pos[:,None],-1e9).softmax(-1)
        y=weights@v.transpose(1,2)
    else:raise ValueError('Architecture forward supports dense/sparse LM, not an invented vision distillation loss')
    return m.out((y.transpose(1,2)*gate.sigmoid()).flatten(-2))


class MultimodalModel(nn.Module):
    def __init__(self,text,visual,image_id=248056,video_id=248057):
        super().__init__();self.text,self.visual=text,visual;self.image_id,self.video_id=image_id,video_id
    def forward(self,ids,types,image_patches=None,image_grid=None,video_patches=None,video_grid=None,stage='sparse',mask=None):
        if mask is not None:
            if mask.shape!=ids.shape or not torch.all((mask==0)|(mask==1)):raise ValueError('Expected binary padding mask matching tokens')
            if not mask.bool().any(1).all():raise ValueError('Empty masked sequence')
            if torch.any((types!=0)&~mask.bool()):raise ValueError('Padding cannot contain modality placeholders')
        if not torch.equal(types==1,ids==self.image_id) or not torch.equal(types==2,ids==self.video_id):raise ValueError('Placeholder and modality types disagree')
        image=None if image_patches is None else self.visual(image_patches,image_grid)[1]
        video=None if video_patches is None else self.visual(video_patches,video_grid)[1]
        embedded=inject_features(ids,self.text.embedding,image,video,self.image_id,self.video_id)
        positions,deltas=multimodal_positions(ids,types,image_grid,video_grid,mask=mask,merge=self.visual.config.spatial_merge_size)
        if mask is None:return self.text_forward(ids,embedded,positions,stage),positions,deltas
        # Reference implementation: each sample's valid tokens form its own
        # sequence. Padding does not advance GDN/PLE state or QSA block indices.
        # Padding logits are deliberately zero; only valid-token logits match
        # the corresponding unpadded forward. No batched-kernel speed claim.
        outputs=[]
        for b,valid in enumerate(mask.bool()):
            logits=self.text_forward(ids[b:b+1,valid],embedded[b:b+1,valid],positions[:,b:b+1,valid],stage)[0]
            outputs.append(logits.new_zeros(ids.shape[1],logits.shape[-1]).index_copy(0,valid.nonzero().flatten(),logits))
        return torch.stack(outputs),positions,deltas

    def text_forward(self,ids,embedded,positions,stage):
        residual=embedded.repeat(1,1,self.text.config.branches)
        for layer in self.text.layers:
            if layer.ple is not None:residual=residual+layer.ple(residual,ids)
            x,write=layer.mix_residual(residual)
            y=attention(layer.mix,x,positions,stage) if isinstance(layer.mix,Attention) else layer.mix(x)
            residual=GatedResidual.add(residual,y,write)
            x,write=layer.moe_residual(residual)
            residual=GatedResidual.add(residual,layer.moe(x)[0],write)
        return self.text.head(self.text.readout(residual))
