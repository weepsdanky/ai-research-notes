"""Four-layer Mac N-gram placement controls using the source-mapped PLE."""
from dataclasses import replace

import torch
from torch import nn

from model import Model, PLE


# Four consecutive primes and the existing 128-row alignment give exactly
# 4096 rows at base 1000, and exactly 2048 rows at base 490.
PLACEMENTS = {
    'none': (),
    'layer1': ((1, 1000),),
    'layer2': ((2, 1000),),
    'layer3': ((3, 1000),),
    'layer4': ((4, 1000),),
    'layer2_3': ((2, 490), (3, 490)),
    'layer2_4': ((2, 490), (4, 490)),
}


class NGramPlacement(Model):
    def __init__(self, config, arm, seed=42):
        if config.layers != 4 or config.ngram_heads != 2:
            raise ValueError('Placement profiles require the four-layer, two-head Mac backbone')
        self.arm = arm
        slots = PLACEMENTS[arm]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # Build the common backbone before any placement-specific allocation.
            super().__init__(replace(config, ngram_layer=0))
            for slot, (layer, base) in enumerate(slots):
                torch.manual_seed(seed + slot + 1)
                ple = PLE(replace(config, ngram_base=base, ngram_layer=layer))
                ple.apply(Model._init)
                nn.init.zeros_(ple.conv.weight)
                self.layers[layer-1].ple = ple
        self.stage('pretrain')

    def hidden(self, ids):
        residual = self.embedding(ids).repeat(1, 1, self.config.branches)
        routers = []
        for layer in self.layers:
            residual, _, router = layer(residual, ids, 'pretrain')
            routers.append(router)
        return self.readout(residual), routers

    def ngram_parameters(self):
        return sum(layer.ple.embedding.ngram_embedding.weight.numel()
                   for layer in self.layers if layer.ple is not None)
