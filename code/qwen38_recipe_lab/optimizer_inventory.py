"""Shape-only optimizer accounting; never materialize official weights or states."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch
from model import Config, Model
from optim import ownership, view


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(plans):
    elements = Counter()
    pieces = Counter()
    for _, parameter, specs in plans:
        covered = 0
        for spec in specs:
            size = view(parameter, spec).numel()
            covered += size
            elements[spec['kind']] += size
            pieces[spec['kind']] += 1
        assert covered == parameter.numel(), 'Logical partition coverage mismatch'
    state = {kind: size*4*(1 if kind == 'muon' else 2) for kind, size in elements.items()}
    return dict(parameter_elements=dict(elements), logical_pieces=dict(pieces),
                all_initialized_fp32_state_tensor_bytes_by_optimizer=state,
                all_initialized_fp32_state_tensor_bytes=sum(state.values()),
                fp32_weight_bytes=4*sum(elements.values()), fp32_gradient_bytes=4*sum(elements.values()))


def official_config():
    source = json.loads(Path('profiles/official.json').read_text())['text_config']
    mapping = dict(vocab_size='vocab_size', eos='eos_token_id', width='hidden_size', layers='num_hidden_layers',
                   heads='num_attention_heads', kv_heads='num_key_value_heads', head_dim='head_dim',
                   gdn_k_heads='linear_num_key_heads', gdn_v_heads='linear_num_value_heads',
                   gdn_k_dim='linear_key_head_dim', gdn_v_dim='linear_value_head_dim', hidden='moe_intermediate_size',
                   experts='num_experts', top_k='num_experts_per_tok', branches='hc_count', rank='hc_lowrank',
                   index_heads='indexer_n_heads', index_dim='indexer_head_dim', block='indexer_compress_ratio',
                   budget='indexer_budget', ngram_base='ngram_vocab_size_base', ngram_heads='heads_per_ngram')
    assert len(source['ple_layer_ids']) == 1
    return Config(**{key: source[value] for key, value in mapping.items()},
                  ngram_layer=source['ple_layer_ids'][0], rotary_dim=int(source['head_dim']*source['partial_rotary_factor']))


def shape_inventory(config):
    # Initialization has no numerical role in a shape inventory. Patch only in this process.
    with torch.device('meta'), patch.object(Model, '_init', staticmethod(lambda module: None)), \
            patch.object(torch.nn.Linear, 'reset_parameters', lambda module: None), \
            patch.object(torch.nn.Embedding, 'reset_parameters', lambda module: None), \
            patch.object(torch.nn.Conv1d, 'reset_parameters', lambda module: None):
        model = Model(config)
    assert all(p.device.type == 'meta' for p in model.parameters())
    result = inventory(ownership(model))
    result['shape_only_meta_parameters'] = True
    return result


def main():
    torch.set_num_threads(1)
    checkpoint = Path('runs/qwen-sparse-paper/last.pt')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    expected_hash = '1b0127c8d93ae879f041f202e789623de837a8fe7e56a557d06da98aa3afcb5e'
    assert digest(checkpoint) == expected_hash and saved['step'] == 8000
    actual_bytes = Counter()
    groups = []
    uninitialized = 0
    for optimizer in saved['optimizers']:
        for group in optimizer['param_groups']:
            parameter = saved['model'][group['name']]
            groups.append((group['name'], parameter, group['pieces']))
            buffers = optimizer['state'].get(group['params'][0], {}).get('pieces', [{} for _ in group['pieces']])
            for spec, state in zip(group['pieces'], buffers, strict=True):
                shape = view(parameter, spec).shape
                if not state:
                    uninitialized += 1
                    continue
                keys = ['momentum'] if spec['kind'] == 'muon' else ['m', 'v']
                assert set(state) == set(keys + ([] if spec['kind'] == 'muon' else ['step']))
                for key in keys:
                    tensor = state[key]
                    assert tensor.shape == shape and tensor.dtype == torch.float32
                    actual_bytes[spec['kind']] += tensor.numel()*tensor.element_size()
    mac = inventory(groups)
    mac.update(actual_state_tensor_bytes_by_optimizer=dict(actual_bytes), actual_state_tensor_bytes=sum(actual_bytes.values()),
               uninitialized_logical_pieces=uninitialized)
    csv_elements = Counter()
    with Path('profiles/mac_optimizer_ownership.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            csv_elements[row['optimizer']] += math.prod(map(int, row['logical_shape'].split('x')))
    assert dict(csv_elements) == mac['parameter_elements'], 'Frozen ownership export drift'
    meta_mac = shape_inventory(Config(**saved['config']))
    assert meta_mac['parameter_elements'] == mac['parameter_elements']
    if uninitialized == 0:
        assert mac['actual_state_tensor_bytes'] == mac['all_initialized_fp32_state_tensor_bytes']
    del saved, groups
    official = shape_inventory(official_config())
    counts = json.loads(Path('profiles/parameter_counts.json').read_text())
    for key, result in [('mac', mac), ('official_text', official)]:
        assert sum(result['parameter_elements'].values()) == counts[key]['parameters'], 'Analytical/module shape count mismatch'
        result['replicated_fp32_weights_gradients_and_all_initialized_state_bytes'] = (
            result['fp32_weight_bytes'] + result['fp32_gradient_bytes'] + result['all_initialized_fp32_state_tensor_bytes'])
    result = dict(mac_trained_text=mac, official_text_shapes_with_declared_mac_ownership=official,
                  sources={name: digest(name) for name in ['model.py', 'optim.py', 'profiles/official.json', 'profiles/mac_optimizer_ownership.tsv', __file__]},
                  checkpoint_sha256=expected_hash,
                  limitations=['Official values are shape-only estimates under our declared optimizer ownership, not verified production ownership or distributed layouts.',
                               'FP32 Muon momentum and Adam first/second moments only; Python step counters, serialization overhead, activations and temporary buffers excluded.',
                               'Text main model only; MTP and vision excluded as in parameter_counts.json.',
                               'Replicated FP32 total is an accounting subtotal, not measured unified memory or production BF16/master/sharded memory.'])
    Path('profiles/optimizer_counts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
