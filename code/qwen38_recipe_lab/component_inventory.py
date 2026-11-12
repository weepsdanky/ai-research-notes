"""Meta-only counts for public visual and incremental MTP component layouts."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from model import Config
from vision import VisionConfig,VisionModel
from mtp import MultiTokenPredictor
from reference import load_reference
from optimizer_inventory import official_config


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def count(module):
    assert all(parameter.device.type == 'meta' for parameter in module.parameters())
    return sum(parameter.numel() for parameter in module.parameters())


def inventory(text, visual, reference):
    with torch.device('meta'):
        ours = VisionModel(visual)
        components = dict(patch_embed=reference['Qwen4ExpVisionPatchEmbed'](visual),
                          pos_embed=nn.Embedding(visual.num_position_embeddings,visual.hidden_size),
                          blocks=nn.ModuleList([reference['Qwen4ExpVisionBlock'](visual) for _ in range(visual.depth)]),
                          merger=reference['Qwen4ExpVisionPatchMerger'](visual))
        for name,module in components.items():
            assert {k:tuple(v.shape) for k,v in getattr(ours,name).state_dict().items()} == {k:tuple(v.shape) for k,v in module.state_dict().items()}
        vision = count(ours)
        assert vision == sum(count(module) for module in components.values())
        mtp = count(MultiTokenPredictor(text))
    d,n,rank = text.width,text.branches,text.rank
    gru = n*d*(2*rank+1)
    write = n*n*d
    attention = d*text.head_dim*(3*text.heads+2*text.kv_heads)+2*text.head_dim
    indexer = (text.index_heads+1)*text.index_dim*d+2*text.index_dim
    moe = 3*d*text.hidden*(text.experts+1)+d*text.experts+d
    formula = 2*d*d+(n+1)*d+attention+indexer+moe+2*(gru+write)+gru
    assert mtp == formula, 'Independent incremental MTP shape formula mismatch'
    return dict(vision_config=asdict(visual),vision_parameters=vision,
                vision_component_parameters={name:count(module) for name,module in components.items()},
                vision_all_state_shapes_match_pinned_upstream_components=True,
                incremental_mtp_parameters=mtp,mtp_formula_verified=True,
                incremental_mtp_excludes_shared_input_embedding_and_output_head=True,
                standalone_draft_input_and_untied_output_replica_elements=2*text.vocab_size*d,
                primary_text_parameters=None)


def main():
    torch.set_num_threads(1)
    lock=json.loads(Path('sources.lock.json').read_text())
    pinned={}
    for name in ['config.official.json','vllm_mtp.py','modeling_qwen4_exp.py','vision_utils.py']:
        path=Path('.cache/references')/name
        expected=next(row['sha256'] for row in lock if row['name']==name)
        assert digest(path)==expected, 'Pinned component source drift: '+name
        pinned[name]=expected
    source=json.loads(Path('profiles/official.json').read_text())
    assert source==json.loads(Path('.cache/references/config.official.json').read_text())

    assert source['text_config']['mtp_num_hidden_layers']==1 and not source['text_config']['mtp_use_dedicated_embeddings']
    assert not source['vision_config']['deepstack_visual_indexes']
    official_visual=VisionConfig(**{k:v for k,v in source['vision_config'].items() if k not in ('model_type','deepstack_visual_indexes')})
    reference=load_reference()
    mac_text=Config(**json.loads(Path('profiles/mac_qwen.json').read_text())['model'])
    result=dict(mac_untrained_component_defaults=inventory(mac_text,VisionConfig(out_hidden_size=mac_text.width),reference),
                published_config_component_shapes=inventory(official_config(),official_visual,reference))
    primary=json.loads(Path('profiles/parameter_counts.json').read_text())
    for key,base in [('mac_untrained_component_defaults','mac'),('published_config_component_shapes','official_text')]:
        entry=result[key];entry['primary_text_parameters']=primary[base]['parameters']
        entry['reconstructed_parameters_with_shared_mtp_vocab_tables']=entry['primary_text_parameters']+entry['vision_parameters']+entry['incremental_mtp_parameters']
    result['pinned_reference_sha256']=pinned
    result['sources']={name:digest(name) for name in ['model.py','mtp.py','vision.py','reference.py','optimizer_inventory.py','profiles/official.json','sources.lock.json',__file__]}
    result['scope']=['Shape-only meta constructors; no forward, large weights, or optimizer states allocated.',
                     'Mac vision/MTP components remain untrained and are not added to the completed main text checkpoint.',
                     'MTP is one public full-attention/MoE draft layer plus fusion/norm/final GR; input/output vocabulary tables counted in main only.',
                     'Standalone vLLM draft creates embedding/head replicas; replica elements are separate runtime storage, not automatically additional unique learned parameters.',
                     'Reconstructed totals are architectural counts under explicit table sharing, not a verified released weight archive inventory or production memory estimate.',
                     'Production MTP/vision optimizer ownership, loss weights, schedule, and complete training recipe remain unknown.']
    Path('profiles/component_counts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
