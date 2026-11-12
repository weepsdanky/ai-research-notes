"""Paper §3.1: logical matrix ownership, independent of fused storage layout.

Qwen-tuned Polar Express: paper-author Canzona, pinned in sources.lock.json.
The production selector/Adam settings remain undisclosed; see the recipe matrix.
"""
import json
import math
from pathlib import Path
import torch

COEFFICIENTS = json.loads(Path(__file__).with_name('polar_coefficients.json').read_text())['coefficients'][:8]


def polar(matrix, *, dtype=torch.bfloat16, eps=1e-14):
    """Canzona order: normalize master grad, cast, then fused addmm iterations.

    float32/float64 are explicit numerical reference variants, not silent fallbacks.
    """
    x = matrix.to(torch.float64 if dtype==torch.float64 else torch.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = torch.nn.functional.normalize(x,p=2,dim=(-2,-1),eps=eps).to(dtype)
    for a, b, c in COEFFICIENTS:
        gram = x @ x.mT
        update = torch.addmm(gram,gram,gram,beta=b,alpha=c)
        x = torch.addmm(x,update,x,beta=a,alpha=1.0)
    return x.mT if transposed else x


def piece(kind, start=None, stop=None, prefix=(), semantic='whole'):
    return dict(kind=kind, start=start, stop=stop, prefix=list(prefix), semantic=semantic)


def view(tensor, spec):
    if spec['start'] is None:
        return tensor
    return tensor[(*spec['prefix'], slice(spec['start'], spec['stop']))]


def head_pieces(heads, dim, *, gated=False):
    specs = []
    for h in range(heads):
        start = h * dim * (2 if gated else 1)
        specs.append(piece('muon', start, start + dim, semantic=f'head{h}'))
        if gated:
            specs.append(piece('adamw', start + dim, start + 2 * dim, semantic=f'gate{h}'))
    return specs


def ownership(model):
    """Modules annotate semantic fused layouts; exclusions never depend on ndim alone."""
    plans = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if hasattr(p, '_recipe_pieces'):
            specs = p._recipe_pieces
        elif '.ngram_embedding.' in name:
            specs = [piece('adam', semantic='ngram_table')]
        elif ('embedding.' in name or name.startswith('head.') or 'indexer.' in name or
              'residual.' in name or 'readout.' in name or 'router.' in name or
              'shared_expert_gate.' in name or p.ndim != 2):
            specs = [piece('adamw')]
        else:
            specs = [piece('muon')]
        plans.append((name, p, specs))
    return plans


class RecipeOptimizer(torch.optim.Optimizer):
    """Per-logical-piece Muon or Adam(W), including gates sharing a fused parameter.

    Non-Muon hyperparameters below are declared Mac choices, not production facts.
    One state dictionary holds all optimizers, so checkpoint resume is atomic.
    """
    def __init__(self, plans, lr, weight_decay=0.1, adam_betas=(0.9, 0.95), adam_eps=1e-8,ns_dtype='bfloat16'):
        groups = [dict(params=[p], name=name, pieces=specs) for name,p,specs in plans]
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay,
                                     adam_betas=adam_betas, adam_eps=adam_eps,ns_dtype=ns_dtype))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closure not used by this recipe')
        for group in self.param_groups:
            p = group['params'][0]
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['pieces'] = [{} for _ in group['pieces']]
            for spec, buffers in zip(group['pieces'], state['pieces'], strict=True):
                parameter, grad = view(p, spec), view(p.grad, spec)
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(f"Nonfinite gradient: {group['name']}/{spec['semantic']}")
                lr, kind = group['lr'], spec['kind']
                decay = 0.0 if kind == 'adam' or parameter.ndim < 2 else group['weight_decay']
                if kind == 'muon':
                    if parameter.ndim != 2:
                        raise ValueError('Each logical Muon piece must be a matrix')
                    if not buffers:
                        buffers['momentum'] = torch.zeros_like(grad)
                    buffers['momentum'].lerp_(grad,0.05)
                    update = polar(grad.lerp(buffers['momentum'],0.95),dtype=getattr(torch,group['ns_dtype']))
                    # Canzona scales in NS dtype, then restores master-gradient dtype.
                    update = (update * (0.2 * math.sqrt(max(parameter.shape)))).to(grad.dtype)
                    parameter.mul_(1-lr*decay)
                    parameter.add_(-lr * update)
                else:
                    b1,b2 = group['adam_betas']
                    if not buffers:
                        buffers.update(m=torch.zeros_like(grad), v=torch.zeros_like(grad), step=0)
                    buffers['step'] += 1
                    buffers['m'].lerp_(grad, 1-b1)
                    buffers['v'].lerp_(grad.square(), 1-b2)
                    t = buffers['step']
                    update = (buffers['m'] / (1-b1**t)) / ((buffers['v'] / (1-b2**t)).sqrt()+group['adam_eps'])
                    parameter.mul_(1-lr*decay).add_(update, alpha=-lr)

    def manifest(self):
        return [dict(name=g['name'], shape=list(g['params'][0].shape), pieces=g['pieces']) for g in self.param_groups]


def make_optimizers(model,stage,name,lr,ns_dtype='bfloat16',weight_decay=0.1):
    model.stage(stage)
    plans=ownership(model)
    if name=='adamw':
        plans=[(name,p,[{**s,'kind':'adamw' if s['kind']=='muon' else s['kind']} for s in specs]) for name,p,specs in plans]
    return [RecipeOptimizer(plans,lr,ns_dtype=ns_dtype,weight_decay=weight_decay)]
