"""Bounded dense inference for Mac residual controls, including both mHC proxies."""
import torch

from attention_residual import DepthHistory
from bounded_dense_inference import BoundedDenseInference
from model import GatedResidual, GDN
from prefill_inference import gdn_prefill


class ResidualInference(BoundedDenseInference):
    def mix(self, layer, inputs, cache):
        if isinstance(layer.mix, GDN):
            return gdn_prefill(layer.mix, inputs, cache)
        return self.attention(layer.mix, inputs, cache)

    def last_hidden(self, ids, cache):
        model = self.model
        mode = 'mhc' if model.variant in ('mhc_static', 'mhc_dynamic') else model.mode
        state = model.embedding(ids)
        if mode == 'gr':
            state = state.repeat(1, 1, model.config.branches)
        elif mode == 'depth':
            state = DepthHistory.start(state)
        elif mode == 'mhc':
            state = state.unsqueeze(-2).expand(-1, -1, model.config.branches, -1)

        for index, layer in enumerate(model.layers):
            slot = cache.setdefault(str(index), {})
            if mode == 'gr':
                inputs, write = layer.mix_residual(state)
                state = GatedResidual.add(state, self.mix(layer, inputs, slot), write)
                inputs, write = layer.moe_residual(state)
                state = GatedResidual.add(state, layer.moe(inputs)[0], write)
            elif mode == 'depth':
                inputs = layer.mix_input(layer.mix_read(state))
                state = state.append(self.mix(layer, inputs, slot), layer.block_size)
                output, _ = layer.moe(layer.moe_input(layer.moe_read(state)))
                state = state.append(output, layer.block_size)
            elif mode == 'mhc':
                mix_map, moe_map = model.maps[2*index], model.maps[2*index+1]
                inputs, write, mix = mix_map(state)
                state = mix_map.update(state, self.mix(layer, layer.mix_input(inputs), slot), write, mix)
                inputs, write, mix = moe_map(state)
                state = moe_map.update(state, layer.moe(layer.moe_input(inputs))[0], write, mix)
            else:
                state = state + self.mix(layer, layer.mix_input(state), slot)
                state = state + layer.moe(layer.moe_input(state))[0]

        if mode == 'depth':
            state = model.depth_read(state)
        elif mode == 'mhc':
            state = state.mean(-2)
        return model.readout(state[:, -1])

    @torch.no_grad()
    def __call__(self, ids, cache):
        if ids.shape[1] < 1:
            raise ValueError('Empty token input')
        chunks = ids.split(self.prefill_chunk, dim=1) if not cache and self.prefill_chunk else (ids,)
        for chunk in chunks:
            last = self.last_hidden(chunk, cache)
        return self.model.head(last)
