"""Audit comparability of two completed recipe variants, without causal claims."""
import json
from pathlib import Path
import torch
from audit_stage import digest


def main():
    torch.set_num_threads(1)
    names={'hf':'qwen-sparse','paper':'qwen-sparse-paper'}
    states={key:torch.load(Path('runs')/name/'last.pt',map_location='cpu',weights_only=True,mmap=True) for key,name in names.items()}
    hf,paper=states['hf'],states['paper']
    sources={name:digest(name) for name in ('model.py','optim.py','polar_coefficients.json','train.py')}
    assert hf['source_hashes']==paper['source_hashes']==sources
    assert hf['step']==paper['step']==8000
    config_diff={k:[hf['config'][k],paper['config'][k]] for k in hf['config'] if hf['config'][k]!=paper['config'][k]}
    assert config_diff=={'score_mode':['hf','paper']}
    keys=('stage','optimizer','ns_dtype','weight_decay','steps','seq','batch','accum','lr','seed','schedule','clip','eval_every','eval_batches')
    assert all(hf['args'][k]==paper['args'][k] for k in keys)
    assert hf['corpora']==paper['corpora'] and torch.equal(hf['rng'],paper['rng'])
    initial_paths={'hf':'runs/qwen-distill-hf/last.pt','paper':'runs/qwen-distill-paper-batch8/last.pt'}
    initial={key:torch.load(path,map_location='cpu',weights_only=True,mmap=True) for key,path in initial_paths.items()}
    backbone=[key for key in initial['hf']['model'] if '.indexer.' not in key]
    assert all(torch.equal(initial['hf']['model'][key],initial['paper']['model'][key]) for key in backbone)
    variants={}
    for key,name in names.items():
        records=[json.loads(line) for line in (Path('runs')/name/'metrics.jsonl').read_text().splitlines()]
        final=next(row for row in reversed(records) if 'val0/lm' in row)
        assert final['step']==8000
        warmup=initial[key]
        variants[key]=dict(final_checkpoint_sha256=digest(Path('runs')/name/'last.pt'),
            distillation_checkpoint_sha256=digest(initial_paths[key]),distillation_steps=warmup['step'],
            distillation_tokens=warmup['step']*warmup['args']['seq']*warmup['args']['batch']*warmup['args']['accum'],
            sparse_tokens=final['tokens_seen'],validation={k:v for k,v in final.items() if k.startswith('val')})
    report=dict(helper_sha256=digest(__file__),training_source_hashes=sources,config_differences=config_diff,
        sparse_phase_args_matched={key:hf['args'][key] for key in keys},corpus_identities_equal=True,final_data_rng_bit_exact=True,
        same_data_order_basis='Same seed, corpus identities and unchanged pure data RNG sampling code, with equal complete step/batch/accum/seq counts and exact final RNG state.',
        distillation_initial_backbone_tensors_bit_exact=len(backbone),variants=variants,
        limitations=['Sparse phases have matched tokens/data/hyperparameters, but indexer initial states and score equations differ.',
            'Distillation token budgets differ by eight times. Whole training histories are not matched.',
            'No causal accuracy/speed/optimizer superiority conclusion follows from this comparison.'])
    Path('runs/paper-hf-comparability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
