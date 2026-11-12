"""Bounded greedy generation for the independent controls' future quality suite.

Uses the separately verified ControlInference cache; does not authorize an
unfinished training checkpoint for formal quality evaluation. The caller must
bind checkpoint/audit/tokenizer/prompt/scorer identities before writing results.
"""
from collections import OrderedDict

import torch


class ControlGreedy:
    def __init__(self, engine, eos, prefill_chunk=128, prefix_slots=4):
        if not isinstance(prefill_chunk, int) or not isinstance(prefix_slots, int) or prefill_chunk < 1 or prefix_slots < 1:
            raise ValueError('Positive chunk and prefix-cache limits required')
        if eos is not None and not 0 <= eos < engine.model.config.vocab_size:
            raise ValueError('EOS must be a model token ID or None')
        self.engine = engine
        self.eos = eos
        self.prefill_chunk = prefill_chunk
        self.prefix_slots = prefix_slots
        self.prefixes = OrderedDict()

    @torch.no_grad()
    def generate(self, prompts, tokens, prefix=(), stopping=None):
        """Equal-length batches only; EOS and optional callback include stop token.

        `stopping(row, output_ids)` must be deterministic and side-effect free.
        Prefix entries are scoped to this engine/model and kept immutable.
        Capacity failures raise before inference; there is no prompt truncation.
        """
        if not prompts or not prompts[0] or not isinstance(tokens, int) or tokens < 1:
            raise ValueError('Nonempty prompts and a positive token budget required')
        length = len(prompts[0])
        if any(len(row) != length for row in prompts):
            raise ValueError('Batch prompts must have equal length; padding is not supported')
        if length + tokens > self.engine.max_context:
            raise ValueError('Context capacity exceeded; preserve the original prompt')
        prefix = tuple(prefix)
        if len(prefix) > length or any(tuple(row[:len(prefix)]) != prefix for row in prompts):
            raise ValueError('Every prompt must match the exact cached prefix')
        device = self.engine.model.embedding.weight.device
        ids = torch.tensor(prompts, device=device, dtype=torch.long)
        if bool(((ids < 0) | (ids >= self.engine.model.config.vocab_size)).any()):
            raise ValueError('Token ID outside the model vocabulary')
        if prefix:
            cached = self.prefixes.pop(prefix, None)
            if cached is None:
                logits, cache = self.engine.prefill(ids[:1, :len(prefix)], self.prefill_chunk)
                cached = logits, cache
            self.prefixes[prefix] = cached
            if len(self.prefixes) > self.prefix_slots:
                self.prefixes.popitem(last=False)
            cache = self.engine.pack([self.engine.fork(cached[1]) for _ in prompts])
            if len(prefix) == length:
                logits = cached[0].expand(len(prompts), -1)
            else:
                logits, cache = self.engine.prefill(ids[:, len(prefix):], self.prefill_chunk, cache)
        else:
            logits, cache = self.engine.prefill(ids, self.prefill_chunk)
        outputs = [[] for _ in prompts]
        reasons = ['max_new_tokens'] * len(prompts)
        done = [False] * len(prompts)
        for step in range(tokens):
            selected = logits.argmax(-1)
            for row, token in enumerate(selected.cpu().tolist()):
                if done[row]:
                    continue
                outputs[row].append(token)
                if token == self.eos:
                    reasons[row] = 'eos'
                    done[row] = True
                elif stopping is not None and stopping(row, outputs[row]):
                    reasons[row] = 'stop_string'
                    done[row] = True
            if all(done) or step + 1 == tokens:
                break
            logits = self.engine.forward(selected[:, None], cache)
        return outputs, reasons
