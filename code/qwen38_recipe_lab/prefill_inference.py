"""Public Torch GDN chunk scan for prefill, recurrent math for token decode.

Training weights/source stay unchanged. This uses a public CPU reference,
not a fused QSA kernel. Floating-point association differs from token scan.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F
from model import Attention, GatedResidual
from inference import Inference
from reference import load_reference


@lru_cache
def chunk_kernel():
    source = Path('.cache/references/modeling_qwen4_exp.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != '3424f1b7a7e4d690c87a43758d791f37cede3fcb7e75f52e80ca498f9b84276e':
        raise ValueError('Unvalidated public GDN reference')
    return load_reference()['torch_chunk_gated_delta_rule']


def gdn_prefill(m, x, cache):
    if x.shape[1] == 1:return m(x, cache)
    c = m.c;b, t, _ = x.shape;kd = c.gdn_k_heads*c.gdn_k_dim
    projected = m.in_proj_qkv(x).transpose(1, 2)
    previous = cache.get('conv')
    history = torch.cat((previous, projected), -1) if previous is not None else F.pad(projected, (3, 0))
    cache['conv'] = history[..., -3:]
    streams = F.silu(m.conv(history)).transpose(1, 2)
    q, k, v = streams.split((kd, kd, c.gdn_v_heads*c.gdn_v_dim), -1)
    q, k = [s.reshape(b, t, c.gdn_k_heads, c.gdn_k_dim).repeat_interleave(c.gdn_v_heads//c.gdn_k_heads, 2) for s in (q, k)]
    q, k = [s.float()*torch.rsqrt(s.float().square().sum(-1, keepdim=True)+1e-6) for s in (q, k)]
    v = v.reshape(b, t, c.gdn_v_heads, c.gdn_v_dim)
    g = -m.A_log.float().exp()*F.softplus(m.in_proj_a(x).float()+m.dt_bias)
    out, cache['state'] = chunk_kernel()(q, k, v, g, m.in_proj_b(x).sigmoid(), chunk_size=64,
                                       initial_state=cache.get('state'), output_final_state=True)
    out = out*torch.rsqrt(out.square().mean(-1, keepdim=True)+1e-6)*m.norm_weight
    return m.out_proj((out*m.in_proj_z(x).reshape_as(out).sigmoid()).flatten(-2))


class PrefillInference(Inference):
    def hidden(self, ids, cache):
        model = self.model
        residual = model.embedding(ids).repeat(1, 1, model.config.branches)
        for i, layer in enumerate(model.layers):
            local = cache.setdefault(str(i), {})
            if layer.ple is not None:residual = residual+layer.ple(residual, ids, local.setdefault('ple', {}))
            x, write = layer.mix_residual(residual);slot = local.setdefault('mix', {})
            y = self.attention(layer.mix, x, slot) if isinstance(layer.mix, Attention) else gdn_prefill(layer.mix, x, slot)
            residual = GatedResidual.add(residual, y, write)
            x, write = layer.moe_residual(residual)
            residual = GatedResidual.add(residual, layer.moe(x)[0], write)
        return residual
