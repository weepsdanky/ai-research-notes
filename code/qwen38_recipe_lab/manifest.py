"""Analytical parameter accounting: never allocate the official-scale weights."""
import json, math
from pathlib import Path


def primes(start, count):
    result=[]
    while len(result)<count:
        if all(start%d for d in range(2,math.isqrt(start)+1)): result.append(start)
        start+=1
    return result


def count(c, ple_layers=1):
    d=c['width']; h=c['hidden']; branches=c['branches']; rank=c['rank']; n=c['layers']
    nq=n//4; ng=n-nq
    kd=c['gdn_k_heads']*c['gdn_k_dim']; vd=c['gdn_v_heads']*c['gdn_v_dim']
    index=(c['index_heads']+1)*c['index_dim']*d+2*c['index_dim']
    gdn=d*(2*kd+3*vd)+2*d*c['gdn_v_heads']+2*c['gdn_v_heads']+4*(2*kd+vd)+c['gdn_v_dim']
    attention=d*c['head_dim']*(3*c['heads']+2*c['kv_heads'])+2*c['head_dim']
    gru=branches*d*(2*rank+1); grwrite=branches*branches*d
    table_rows=math.ceil(sum(primes(c['ngram_base'],2*c['ngram_heads']))/128)*128
    table=table_rows*(d//(2*c['ngram_heads']))
    experts=3*d*h*c['experts']; shared=3*d*h; routing=d*c['experts']+d
    parts=dict(input_embedding=c['vocab_size']*d,output_head=c['vocab_size']*d,
        gdn=ng*gdn,attention=nq*attention,indexer=nq*index,
        gated_residual=2*n*(gru+grwrite)+gru,
        routed_experts=n*experts,shared_experts=n*shared,router_and_shared_gate=n*routing,
        ngram_table=ple_layers*table,ple_without_table=ple_layers*(branches*d*d+d*d+7*branches*d))
    total=sum(parts.values())
    active=total-parts['input_embedding']+d-parts['ngram_table']+ple_layers*d
    active-=n*3*d*h*(c['experts']-c['top_k'])
    return dict(parameters=total,components=parts,
        active_parameters_per_token_dense=active-parts['indexer'],
        active_parameters_per_token_qsa=active,
        active_definition='logical parameter touches, includes full vocab output projection, excludes unselected expert and lookup rows; not a compute/memory measurement',
        weight_bytes_fp32=4*total,weight_bytes_bfloat16=2*total,
        excludes=['vision','MTP','production sharding/quantization/state layouts'])


def main():
    mac=json.loads(Path('profiles/mac_qwen.json').read_text())['model']
    official=json.loads(Path('profiles/official.json').read_text())['text_config']
    mapping=dict(vocab_size='vocab_size',width='hidden_size',hidden='moe_intermediate_size',layers='num_hidden_layers',
        heads='num_attention_heads',kv_heads='num_key_value_heads',head_dim='head_dim',
        gdn_k_heads='linear_num_key_heads',gdn_v_heads='linear_num_value_heads',gdn_k_dim='linear_key_head_dim',gdn_v_dim='linear_value_head_dim',
        experts='num_experts',top_k='num_experts_per_tok',branches='hc_count',rank='hc_lowrank',
        index_heads='indexer_n_heads',index_dim='indexer_head_dim',ngram_base='ngram_vocab_size_base',ngram_heads='heads_per_ngram')
    result={'mac':count(mac),'official_text':count({k:official[v] for k,v in mapping.items()},len(official['ple_layer_ids']))}
    # Internal arithmetic invariant; module-level comparison is performed separately.
    assert result['mac']['parameters']==32978962
    Path('profiles/parameter_counts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
