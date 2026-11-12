"""Read-only finite/sign diagnostics on actual frozen backbone activations.

This locates dead ReLU regions; it does not establish the optimizer's causal
history, repair a failed run, or change the distillation acceptance criteria.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from model import Config, Indexer, Model, rope
from train import Corpus


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    torch.set_num_threads(1)
    paths = {name: Path('runs') / directory / 'last.pt' for name, directory in (
        ('initial', 'qwen-cpt'), ('paper', 'qwen-distill-paper'),
        ('paper_lr1e4', 'qwen-distill-paper-lr1e4'), ('hf', 'qwen-distill-hf'),
        ('paper_batch8', 'qwen-distill-paper-batch8'))}
    hashes = {name: digest(path) for name, path in paths.items()}
    saved = torch.load(paths['initial'], map_location='cpu', weights_only=True, mmap=True)
    reference_config = saved['config']
    reference_backbone = {key: value for key, value in saved['model'].items() if '.indexer.' not in key}
    config = Config(**reference_config)
    model = Model(config).eval()
    model.load_state_dict(saved['model'])
    captured = []
    handles = [module.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
               for module in model.modules() if isinstance(module, Indexer)]
    assert len(handles) == 1
    inputs = {}
    data = {}
    with torch.no_grad():
        for name in ('owt', 'shakespeare'):
            corpus = Corpus('data/' + name, 128)
            rng = torch.Generator().manual_seed(1234)
            inputs[name] = []
            data[name] = dict(meta=corpus.meta, validation_batch_ids_sha256=[])
            for _ in range(8):
                ids, _ = corpus.batch('val', 1, 128, rng, 'cpu')
                data[name]['validation_batch_ids_sha256'].append(hashlib.sha256(ids.numpy().tobytes()).hexdigest())
                captured.clear()
                state = model.embedding(ids).repeat(1, 1, config.branches)
                for layer in model.layers:
                    state, _, _ = layer(state, ids, 'distill')
                assert len(captured) == 1
                inputs[name].append(captured[0])
    for handle in handles:
        handle.remove()
    del model, saved
    results = {}
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert {key: value for key, value in checkpoint['config'].items() if key != 'score_mode'} == {key: value for key, value in reference_config.items() if key != 'score_mode'}
        assert all(torch.equal(value, checkpoint['model'][key]) for key, value in reference_backbone.items())
        c = Config(**checkpoint['config'])
        prefix = 'layers.3.mix.indexer.'
        indexer = Indexer(c).eval()
        indexer.load_state_dict({key.removeprefix(prefix): value for key, value in checkpoint['model'].items() if key.startswith(prefix)})
        del checkpoint
        corpora = {}
        with torch.no_grad():
            for corpus, batches in inputs.items():
                positive = dot_count = zero = score_count = dead = queries = 0
                dot_min, dot_max, score_error = math.inf, -math.inf, 0.
                mean_energy = []
                for x in batches:
                    b, t, _ = x.shape
                    blocks = t // c.block
                    scores, valid, _, _ = indexer(x)
                    q, k = indexer.qk(x).split((c.index_heads * c.index_dim, c.index_dim), -1)
                    q = rope(indexer.qnorm(q.reshape(b, t, c.index_heads, c.index_dim)), torch.arange(t), c.rotary_dim, c.rope_theta)
                    k = k[:, :blocks*c.block].reshape(b, blocks, c.block, c.index_dim).float().mean(2)
                    k = rope(indexer.knorm(k)[:, :, None], torch.arange(blocks)*c.block, c.rotary_dim, c.rope_theta)[:, :, 0]
                    dot = torch.einsum('bthd,bsd->bths', q, k)
                    mask = valid[None].expand_as(scores)
                    values = dot[valid[None, :, None, :].expand_as(dot)]
                    assert torch.isfinite(values).all()
                    manual = dot.relu().sum(2)
                    if c.score_mode == 'hf':
                        manual = manual / math.sqrt(c.index_dim)
                    torch.testing.assert_close(scores[mask], manual[mask], rtol=0, atol=0)
                    score_error = max(score_error, float((scores[mask]-manual[mask]).abs().max()))
                    positive += int((values > 0).sum()); dot_count += values.numel()
                    zero += int((scores[mask] == 0).sum()); score_count += int(mask.sum())
                    usable = valid.sum(-1) >= 2
                    dead_queries = ((dot <= 0) | ~valid[None, :, None, :]).all(-1).all(-1)
                    dead += int(dead_queries[:, usable].sum()); queries += int(usable.sum()) * b
                    dot_min = min(dot_min, float(values.min())); dot_max = max(dot_max, float(values.max()))
                    mean_energy.append(float(x.mean(1).square().sum(-1).mean() / x.square().sum(-1).mean()))
                corpora[corpus] = dict(batches=len(batches), positive_pre_relu_dot_fraction=positive/dot_count,
                    zero_valid_score_fraction=zero/score_count, fully_dead_eligible_query_fraction=dead/queries,
                    eligible_queries=queries, pre_relu_dot_min=dot_min, pre_relu_dot_max=dot_max,
                    input_mean_energy_fraction=sum(mean_energy)/len(mean_energy),
                    manual_scores_max_abs_error=score_error)
        if name != 'initial':
            label = {'paper': 'paper', 'paper_lr1e4': 'paper-lr1e4', 'hf': 'hf', 'paper_batch8': 'paper-batch8'}[name]
            audit = json.loads(Path(f'runs/audit-qwen-{label}.json').read_text())['corpora']
            for corpus, values in corpora.items():
                assert abs(1-values['zero_valid_score_fraction']-audit['data/'+corpus]['nonzero_score_fraction']) < 1e-7
        results[name] = dict(score_mode=c.score_mode, norm_gain_ranges={key: [float((1+module.weight).min().detach()), float((1+module.weight).max().detach())]
            for key, module in [('q', indexer.qnorm), ('k', indexer.knorm)]}, corpora=corpora)
    assert hashes == {name: digest(path) for name, path in paths.items()}
    report = dict(checkpoints_sha256=hashes, data=data, all_variant_non_indexer_tensors_bit_exact_to_CPT=True, source_sha256={file: digest(file) for file in ('model.py', __file__)},
        scope='CPU FP32; eight fixed actual frozen-CPT validation batches per corpus; no vocabulary head, training, checkpoint writes or synthetic hidden activations.',
        interpretation='Dead queries have all eligible pre-ReLU dot products <=0, so the score derivative is zero there. This diagnoses the saved state, not a causal optimizer history or a training repair.', results=results)
    Path('runs/indexer-collapse-diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
