"""CPU FP32 control gradients with serializable forward/CE/layer boundaries.

No optimizer step is taken here. Partial work never counts as a training update.
Each backward layer reproduces reentrant checkpoint's local backward; the CE
uses the frozen reverse chunk-backward and forward loss-sum order. Snapshot
returns tensor views like Module.state_dict: serialize immediately or deepcopy.
"""
from dataclasses import asdict
import torch
from torch.nn import functional as F
from train_qwen35_controls import balance_loss


class StagedControlUpdate:
    def __init__(self, model, x, targets, *, binding, chunk=8, balance_weight=.001, state=None):
        if not binding or chunk<1 or not model.training:
            raise ValueError('An immutable owner binding, positive chunk and training mode are required')
        if x.dtype!=torch.long or targets.dtype!=torch.long or x.shape!=targets.shape or x.numel()==0:
            raise ValueError('Aligned nonempty token inputs and targets required')
        if x.device.type!='cpu' or targets.device.type!='cpu' or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters()):
            raise ValueError('This exact continuation path is CPU FP32 only')
        if model.head.weight.data_ptr()==model.embedding.weight.data_ptr():
            raise ValueError('Independent head and embedding required')
        self.model=model;self.boundary=None
        spec=dict(version=1,binding=binding,config=asdict(model.config),arm=model.arm,chunk=chunk,balance_weight=balance_weight)
        if state is None:
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Start with cleared gradients')
            self.s=dict(spec=spec,x=x.detach(),targets=targets.detach(),phase='embedding_forward',inputs=[],routers=[],ce_done=0,losses=[],grads={},units=0)
        else:
            if state['spec']!=spec or not torch.equal(x,state['x']) or not torch.equal(targets,state['targets']):
                raise ValueError('Owner/config/data continuation mismatch')
            if state['phase'] not in ['embedding_forward','layer_forward','norm_forward','head','aux','norm_backward','layer_backward','embedding_backward','done']:
                raise ValueError('Unknown staged checkpoint phase')
            self.s=state
            params=dict(model.named_parameters())
            if not state['grads'].keys()<=params.keys():raise ValueError('Unknown saved parameter gradient')
            for name,p in params.items():
                g=state['grads'].get(name)
                if g is not None and (g.shape!=p.shape or g.dtype!=p.dtype or g.device.type!='cpu'):raise ValueError('Saved gradient shape/dtype mismatch')
                p.grad=g
            if self.s['phase']=='head':
                self.boundary=self.s['hidden'].detach().requires_grad_(True)
                self.boundary.grad=self.s.get('hidden_adjoint')
        count=(targets.numel()+chunk-1)//chunk
        if not 0<=self.s['ce_done']<=count or len(self.s['losses'])!=self.s['ce_done']:
            raise ValueError('Invalid CE progress')

    @property
    def done(self):return self.s['phase']=='done'

    def snapshot(self):
        state=dict(self.s)
        state['grads']={name:p.grad.detach() for name,p in self.model.named_parameters() if p.grad is not None}
        if self.boundary is not None:state['hidden_adjoint']=None if self.boundary.grad is None else self.boundary.grad.detach()
        return state

    def advance(self):
        """One forward layer, CE chunk, auxiliary VJP, or backward layer."""
        if self.done:return False
        s=self.s;m=self.model;phase=s['phase']
        if phase=='embedding_forward':
            with torch.no_grad():s['inputs']=[m.embedding(s['x']).detach()]
            s['phase']='layer_forward'
        elif phase=='layer_forward':
            index=len(s['routers'])
            with torch.no_grad():out,router=m.layers[index](s['inputs'][-1])
            s['inputs'].append(out.detach());s['routers'].append(router.detach())
            if len(s['routers'])==len(m.layers):s['phase']='norm_forward'
        elif phase=='norm_forward':
            with torch.no_grad():s['hidden']=m.norm(s['inputs'][-1]).detach()
            self.boundary=s['hidden'].detach().requires_grad_(True);s['phase']='head'
        elif phase=='head':
            labels=s['targets'].reshape(-1);flat=self.boundary.reshape(-1,self.boundary.shape[-1])
            start=(len(labels)-1)//s['spec']['chunk']*s['spec']['chunk']-s['ce_done']*s['spec']['chunk']
            with torch.enable_grad():
                loss=F.cross_entropy(F.linear(flat[start:start+s['spec']['chunk']],m.head.weight).float(),labels[start:start+s['spec']['chunk']],reduction='sum')
                if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite complete-vocabulary CE')
                (loss/labels.numel()).backward()
            s['losses'].append(loss.detach());s['ce_done']+=1
            if s['ce_done']==(len(labels)+s['spec']['chunk']-1)//s['spec']['chunk']:
                total=s['hidden'].new_zeros((),dtype=torch.float32)
                for value in reversed(s['losses']):total=total+value
                s['lm']=total/labels.numel();s['hidden_adjoint']=self.boundary.grad.detach();self.boundary=None;s['phase']='aux'
        elif phase=='aux':
            with torch.enable_grad():
                routers=[value.detach().requires_grad_(True) for value in s['routers']]
                balance=balance_loss(routers,m.config.experts,m.config.top_k)
                adjoints=torch.autograd.grad(balance,routers,grad_outputs=balance.new_tensor(s['spec']['balance_weight']))
            s['balance']=balance.detach();s['router_adjoints']=[value.detach() for value in adjoints];s['phase']='norm_backward'
        elif phase=='norm_backward':
            with torch.enable_grad():
                leaf=s['inputs'][-1].detach().requires_grad_(True);out=m.norm(leaf)
                torch.autograd.backward(out,s['hidden_adjoint'])
            s['adjoint']=leaf.grad.detach();s['backward_index']=len(m.layers)-1;s['phase']='layer_backward'
        elif phase=='layer_backward':
            index=s['backward_index']
            with torch.enable_grad():
                leaf=s['inputs'][index].detach().requires_grad_(True);out,router=m.layers[index](leaf)
                torch.autograd.backward((out,router),(s['adjoint'],s['router_adjoints'][index]))
            s['adjoint']=leaf.grad.detach();s['backward_index']-=1
            if s['backward_index']<0:s['phase']='embedding_backward'
        elif phase=='embedding_backward':
            with torch.enable_grad():torch.autograd.backward(m.embedding(s['x']),s['adjoint'])
            s['phase']='done'
        s['units']+=1
        return True
