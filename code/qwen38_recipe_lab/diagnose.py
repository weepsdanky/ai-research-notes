"""Per-layer expert and actual QSA pruning audit on a saved local checkpoint."""
import argparse, json
from pathlib import Path
import torch
from model import Config, Model, MoE, Indexer
from train import Corpus


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--out',required=True)
    p.add_argument('--data',nargs='+',default=['data/owt','data/shakespeare'])
    p.add_argument('--batches',type=int,default=8)
    a=p.parse_args(); torch.set_num_threads(1)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=Model(Config(**saved['config'])).eval(); model.load_state_dict(saved['model'])
    c=model.config; seq=saved['args']['seq']; stage='sparse' if saved['args']['stage']=='sparse' else 'pretrain'
    captures={}; handles=[]
    for name,module in model.named_modules():
        if isinstance(module,(MoE,Indexer)):
            def capture(m,inputs,outputs,name=name): captures[name]=(inputs,outputs)
            handles.append(module.register_forward_hook(capture))
    corpora={}
    for directory in a.data:
        data=Corpus(directory,seq); rng=torch.Generator().manual_seed(1234)
        usage={}; shared={}; pruning=[]
        for _ in range(a.batches):
            ids,_=data.batch('val',1,seq,rng,'cpu'); captures.clear()
            state=model.embedding(ids).repeat(1,1,c.branches)
            for layer in model.layers: state,_,_=layer(state,ids,stage)
            for name,(inputs,outputs) in captures.items():
                module=model.get_submodule(name)
                if isinstance(module,MoE):
                    counts=torch.bincount(outputs[1].topk(c.top_k,-1).indices.flatten(),minlength=c.experts)
                    usage[name]=usage.get(name,torch.zeros_like(counts))+counts
                    shared.setdefault(name,[]).append(module.shared_expert_gate(inputs[0]).sigmoid().mean().item())
                else:
                    scores,valid,selected,tokens=outputs
                    eligible=valid.sum(-1); selected_blocks=(selected>=0).sum(-1)
                    causal=torch.arange(seq)+1
                    actual=(tokens>=0).sum(-1)
                    pruning.append(dict(pruned_query_fraction=(actual<causal).float().mean().item(),
                        selected_at_least_two_blocks_fraction=(selected_blocks>=2).float().mean().item(),
                        mean_selected_tokens=actual.float().mean().item(),max_selected_tokens=int(actual.max()),
                        mean_causal_tokens=causal.float().mean().item(),max_eligible_blocks=int(eligible.max())))
        experts={}
        for name,counts in usage.items():
            fraction=counts.float()/counts.sum()
            experts[name]=dict(counts=counts.tolist(),slot_fraction=fraction.tolist(),
                unused_experts=int((counts==0).sum()),max_slot_fraction=float(fraction.max()),
                normalized_entropy=float(-(fraction*fraction.clamp_min(1e-20).log()).sum()/torch.log(torch.tensor(float(c.experts)))),
                mean_shared_gate=sum(shared[name])/len(shared[name]))
        corpora[directory]=dict(experts=experts,qsa=pruning)
    for h in handles: h.remove()
    total=sum(p.numel() for p in model.parameters()); embedding=model.embedding.weight.numel()
    ngram=sum(m.ngram_embedding.weight.numel() for m in model.modules() if hasattr(m,'ngram_embedding'))
    inactive_experts=sum(sum(p.numel() for p in m.experts[0].parameters())*(c.experts-c.top_k) for m in model.modules() if isinstance(m,MoE))
    indexer=sum(p.numel() for m in model.modules() if isinstance(m,Indexer) for p in m.parameters())
    active=total-embedding+c.width-ngram+c.width-inactive_experts-(indexer if stage=='pretrain' else 0)
    state_bytes={}
    def count(value):
        if isinstance(value,torch.Tensor): state_bytes[str(value.dtype)]=state_bytes.get(str(value.dtype),0)+value.numel()*value.element_size()
        elif isinstance(value,dict):
            for item in value.values(): count(item)
        elif isinstance(value,(list,tuple)):
            for item in value: count(item)
    count(saved['optimizers'])
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),step=saved['step'],stage=stage,
        parameters=total,embedding_parameters=embedding,ngram_table_parameters=ngram,
        logical_active_parameters_per_token=active,
        active_count_definition='all output projection + dense body + top-k expert parameters + one input embedding row + one ngram row per head/order; no memory/throughput claim; indexer omitted for pretrain',
        optimizer_state_bytes_by_dtype=state_bytes,corpora=corpora)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
