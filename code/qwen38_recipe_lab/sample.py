"""Actual generation using the official locked tokenizer; keeps raw token IDs."""
import argparse,json,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Config,Model
from train import device_for

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('--prompt',default='ROMEO:')
    p.add_argument('--tokens',type=int,default=100); p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--device',choices=['auto','mps','cpu'],default='mps'); p.add_argument('--out',required=True)
    a=p.parse_args()
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    device=device_for(a.device)
    model=Model(Config(**c['config'])).to(device).eval(); model.load_state_dict(c['model'])
    tokenizer=Tokenizer.from_file('.cache/references/tokenizer.json')
    stage='sparse' if c['args']['stage']=='sparse' else 'pretrain'
    prompt_ids=tokenizer.encode(a.prompt,add_special_tokens=False).ids
    ids=prompt_ids.copy(); generated=[]
    torch.manual_seed(42)
    for _ in range(a.tokens):
        x=torch.tensor([ids[-c['args']['seq']:]],device=device)
        logits=model(x,stage)[0][0,-1]
        token=torch.multinomial((logits/a.temperature).softmax(-1).cpu(),1).item()
        generated.append(token)
        if token==model.config.eos: break
        ids.append(token)
    with Path(a.checkpoint).open('rb') as stream:
        checkpoint_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(prompt=a.prompt,seed=42,temperature=a.temperature,stage=stage,ids=generated,
                decoded=tokenizer.decode(prompt_ids+generated,skip_special_tokens=False),
                step=c['step'],target_steps=c['args']['steps'],phase_complete=c['step']==c['args']['steps'],
                context_mode='recompute sliding window',context_tokens=c['args']['seq'],
                unused_token_ids=[i for i in generated if tokenizer.id_to_token(i) is None],
                checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=checkpoint_sha256)
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report['decoded'])

if __name__=='__main__': main()
