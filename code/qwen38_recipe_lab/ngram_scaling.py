"""Construct the locked Mac N-gram table capacity controls."""
from dataclasses import replace
import hashlib

import torch
from torch import nn

from model import GDN, Model, PLE
from ngram_placement import NGramPlacement


class NGramScaling(Model):
    # Placement and scaling use the same dense pretraining forward and table count.
    hidden = NGramPlacement.hidden
    ngram_parameters = NGramPlacement.ngram_parameters

    def __init__(self, config, arm, entry, position, seed):
        table = entry['table']
        active = table['rows'] > 0
        config = replace(config, experts=entry['experts'],
                         ngram_layer=position if active else 0,
                         ngram_base=table['ngram_base'] if active else config.ngram_base)
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(config)
            self._initialize_shared(seed)
        self.stage('pretrain')
        assert self.ngram_parameters() == table['parameters']
        if active:
            ple = self.layers[position-1].ple
            assert ple.embedding.sizes == table['primes']
            assert ple.embedding.ngram_embedding.num_embeddings == table['rows']
            adapter = sum(p.numel() for p in ple.parameters()) - self.ngram_parameters()
            assert adapter == entry['added_ple_adapter_parameters']

    @torch.no_grad()
    def _initialize_shared(self, seed):
        # Name-local initialization prevents extra experts changing shared weights.
        for name, module in self.named_modules():
            material = hashlib.sha256(f'{seed}:{name}'.encode()).digest()
            rng = torch.Generator().manual_seed(int.from_bytes(material[:8], 'little') % (2**63))
            if isinstance(module, GDN):
                module.A_log.uniform_(0.01, 16, generator=rng).log_()
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                if name.endswith('.moe.router'):
                    # Row-wise draws guarantee that overlapping router rows match.
                    for row in module.weight:
                        row.normal_(std=0.02, generator=rng)
                elif name.endswith('.ple.conv'):
                    module.weight.zero_()
                else:
                    module.weight.normal_(std=0.02, generator=rng)
