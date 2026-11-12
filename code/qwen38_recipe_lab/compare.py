"""Verify matched initialization/data budgets and report the bounded controls."""
import gc, hashlib, json
from pathlib import Path
import torch
from model import Config, Model


def initial_hash(config):
    torch.manual_seed(42)
    model=Model(Config(**config))
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def metrics(name):
    return [json.loads(s) for s in (Path('runs')/name/'metrics.jsonl').read_text().splitlines()]


def main():
    torch.set_num_threads(1); records={}; hashes={}; rngs=[]; sources=[]
    for name in ('qwen-control-muon','qwen-control-adamw','qwen-control-ungated'):
        saved=torch.load(Path('runs')/name/'last.pt',map_location='cpu',weights_only=True)
        args=saved['args']
        assert saved['step']==args['steps']==256
        assert (args['seed'],args['seq'],args['batch'],args['accum'],args['lr'],args['schedule'])==(42,128,1,1,.0003,'constant')
        rngs.append(saved['rng'].clone()); sources.append(saved['source_hashes'])
        hashes[name]=initial_hash(saved['config'])
        rows=metrics(name); first=next(r for r in rows if r['step']==0)
        last=next(r for r in reversed(rows) if 'val0/lm'in r)
        grads=[r['grad_norm'] for r in rows if 'grad_norm'in r]
        records[name]=dict(steps=saved['step'],tokens=32768,initial_lm=first['val0/lm'],final_lm=last['val0/lm'],
            max_grad_norm=max(grads),nonfinite_steps=sum(not __import__('math').isfinite(g) for g in grads),
            optimizer=args['optimizer'],gr_mode=args['gr_mode'])
        del saved;gc.collect()
    assert len(set(hashes.values()))==1,'Controls must start from identical weights, including gate ablation'
    assert all(torch.equal(rngs[0],r) for r in rngs[1:])
    assert all(sources[0]==s for s in sources[1:])
    dense=torch.load('runs/qwen-dense-control/last.pt',map_location='cpu',weights_only=True)
    assert dense['step']==dense['args']['steps']==200
    dense_rng=dense['rng'].clone(); dense_sources=dense['source_hashes']; dense_corpora=dense['corpora']; dense_args=dense['args']
    del dense; gc.collect()
    sparse=torch.load('runs/comparisons/qwen-sparse-step200.pt',map_location='cpu',weights_only=True)
    assert sparse['step']==200 and sparse['source_hashes']==dense_sources
    assert torch.equal(sparse['rng'],dense_rng) and sparse['corpora']==dense_corpora
    for key in ('seed','seq','batch','accum','lr','clip','weight_decay','ns_dtype','optimizer'):
        assert sparse['args'][key]==dense_args[key],key
    assert dense_args['schedule']=='constant' and sparse['args']['stage']=='sparse'
    dense_row=next(r for r in metrics('qwen-dense-control') if r.get('step')==200 and 'val0/lm'in r)
    sparse_row=next(r for r in metrics('qwen-sparse') if r.get('step')==200 and 'val0/lm'in r)
    result=dict(status='bounded Mac controls, not official full-scale ablations',initial_parameter_hashes=hashes,
        initialization_identical=True,data_rng_identical=True,algorithm_sources_identical=True,
        controls=records,dense_qsa=dict(steps=200,tokens=25600,lr=.000025,
            backbone_initialization_identical_evidence='runs/qwen-distillation-audit.json; CPT weights unchanged except indexer',
            data_rng_identical=True,dense={k:v for k,v in dense_row.items() if k.startswith('val')},
            sparse={k:v for k,v in sparse_row.items() if k.startswith('val')},
            interpretation='independent dense LM continuation vs sparse LM+KL continuation; different objectives, same backbone starting weights and token sequence; not same-weight mask toggling'))
    Path('runs/qwen-controls-comparison.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
