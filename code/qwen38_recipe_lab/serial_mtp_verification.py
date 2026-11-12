"""Speculative decoding with token-by-token target verification."""

import torch

from speculative_mtp import GreedyMTP


class SerialTargetMTP(GreedyMTP):
    """Use the same cached decode operation as the single-token target oracle."""

    def generate(self, prompt, max_new_tokens=64, draft_tokens=4, eos_token_ids=()):
        self.target_forward_calls = 0
        result = super().generate(prompt, max_new_tokens, draft_tokens, eos_token_ids)
        # Last-position-only prefill bypasses _target in GreedyMTP.
        result['target_forward_calls'] = self.target_forward_calls + int(
            self.prefill_last_only and max_new_tokens > 0
        )
        return result

    def _target(self, ids, cache):
        if not cache or ids.shape[1] == 1:
            self.target_forward_calls += 1
            return super()._target(ids, cache)
        target = super()._target
        self.target_forward_calls += ids.shape[1]
        logits, hidden = zip(*(target(token, cache) for token in ids.split(1, dim=1)), strict=True)
        return torch.cat(logits, dim=1), torch.cat(hidden, dim=1)
