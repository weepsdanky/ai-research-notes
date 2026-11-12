"""Original mHC scalar residual operators for a declared Mac Table 5 proxy."""
import torch
from torch import nn

from optim import piece
from residual_controls import ResidualControl


def sinkhorn(logits, iterations=20):
    """Column then row normalization, carried out in log space."""
    log_weights = logits.float()
    for _ in range(iterations):
        log_weights = log_weights - log_weights.logsumexp(-2, keepdim=True)
        log_weights = log_weights - log_weights.logsumexp(-1, keepdim=True)
    return log_weights.exp()


class ManifoldHyperConnection(nn.Module):
    def __init__(self, width, branches, read_branch, *, dynamic):
        super().__init__()
        if not (width > 0 and branches > 1 and 0 <= read_branch < branches):
            raise ValueError('Invalid widened-stream dimensions or read branch')
        self.width, self.branches = width, branches
        read_bias = torch.full((branches,), -8.0)
        read_bias[read_branch] = 8.0
        self.read_bias = nn.Parameter(read_bias)
        self.write_bias = nn.Parameter(torch.zeros(branches))
        self.mix_bias = nn.Parameter(8*torch.eye(branches))
        self.mix_bias._recipe_pieces = [piece('adamw', semantic='mhc_gate')]
        self.dynamic = dynamic
        if dynamic:
            sizes = (branches, branches, branches*branches)
            self.projections = nn.ModuleList(nn.Linear(width*branches, size, bias=False) for size in sizes)
            self.alpha = nn.Parameter(torch.full((3,), 0.01))
            for projection in self.projections:
                nn.init.normal_(projection.weight, std=0.02)
                projection.weight._recipe_pieces = [piece('adamw', semantic='mhc_gate')]

    def maps(self, residual):
        if residual.shape[-2:] != (self.branches, self.width):
            raise ValueError('Expected branch-first residual')
        shape = residual.shape[:-2]
        biases = (self.read_bias, self.write_bias, self.mix_bias.flatten())
        if self.dynamic:
            flat = residual.float().flatten(-2)
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            read, write, mix = [
                bias + self.alpha[i] * projection(normalized.to(projection.weight.dtype)).float()
                for i, (bias, projection) in enumerate(zip(biases, self.projections, strict=True))
            ]
        else:
            read, write, mix = [bias.expand(*shape, bias.numel()) for bias in biases]
        return read.sigmoid(), 2*write.sigmoid(), sinkhorn(mix.unflatten(-1, (self.branches, self.branches)))

    def forward(self, residual):
        read, write, mix = self.maps(residual)
        value = (read[..., None] * residual.float()).sum(-2).to(residual.dtype)
        return value, write, mix

    @staticmethod
    def update(residual, output, write, mix):
        return (mix @ residual.float() + write[..., None]*output.float().unsqueeze(-2)).to(residual.dtype)


class MHCResidualControl(ResidualControl):
    """Reuse the Table 6 pre-norm backbone and add branch maps before each sublayer."""
    def __init__(self, config, arm, seed=42):
        if arm not in ('mhc_static', 'mhc_dynamic'):
            raise ValueError('Unknown mHC control arm')
        super().__init__(config, 'prenorm', seed)
        self.variant = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1)
            self.maps = nn.ModuleList(
                ManifoldHyperConnection(config.width, config.branches, index % config.branches,
                                         dynamic=arm == 'mhc_dynamic')
                for index in range(2*config.layers))

    def hidden(self, ids):
        state = self.embedding(ids).unsqueeze(-2).expand(-1, -1, self.config.branches, -1)
        routers = []
        for index, layer in enumerate(self.layers):
            attn_map, moe_map = self.maps[2*index], self.maps[2*index+1]
            value, write, mix = attn_map(state)
            state = attn_map.update(state, layer.mix(layer.mix_input(value)), write, mix)
            value, write, mix = moe_map(state)
            output, router = layer.moe(layer.moe_input(value))
            state = moe_map.update(state, output, write, mix)
            routers.append(router)
        return self.readout(state.mean(-2)), routers
