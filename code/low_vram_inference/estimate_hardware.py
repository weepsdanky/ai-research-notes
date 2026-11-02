"""Uncalibrated single-request, layer-split hardware scenarios; not benchmarks.

GB and TFLOP are decimal. Input context is occupied prompt tokens, model is loaded,
prefix reuse/speculation are off, and weights/KV/compute stay on accelerator devices.
"""
import argparse
import json
import math
from pathlib import Path

MODEL_BYTES = 155_095_240_320
EXPERT_BYTES = 43 * 3 * 1088 * 2**20
EXPERT_ACTIVE_BYTES = EXPERT_BYTES * 6 / 256
# Conservative proxy: includes embedding tables/metadata not fully read per token.
WEIGHT_DECODE_BYTES = EXPERT_ACTIVE_BYTES + MODEL_BYTES - EXPERT_BYTES
STATE_BYTES = (1.31 + 10 + .33) * 2**20
DEVICES = {
    'two_spark_layer_split': {'bandwidth_GB_s': 273, 'dense_fp4_reference_TF_s': 500,
                            'base_other_decode_s': .020, 'budget_cny': 110_000},
    'two_pro6000_layer_split': {'bandwidth_GB_s': 1792, 'dense_fp4_reference_TF_s': 2000,
                              'base_other_decode_s': .005, 'budget_cny': 500_000},
}
# Explicit sensitivity inputs, not empirically fitted confidence intervals.
SCENARIOS = {'slow': (.25, .05, 2), 'middle': (.40, .10, 1), 'fast': (.60, .20, .5)}


def pad256(n):
    return math.ceil(n / 256) * 256


def cache(context, ubatch=32):
    csa_cells = pad256(math.ceil(context / 4))
    hca_cells = pad256(math.ceil(context / 128))
    return {'raw': 43 * pad256(min(context, 128 + ubatch)) * 512 * 2,
            'csa': 21 * csa_cells * 512 * 2,
            'hca': 20 * hca_cells * 512 * 2,
            'lid': 21 * csa_cells * 128 * 2, 'state': STATE_BYTES}


def decode_bytes(context, ubatch=512):
    c = cache(context, ubatch)
    # Graph-visible concat read+write, one LID scan, and two K/V reads.
    # This is a traffic scenario, NOT measured DRAM traffic; fusion/L2 may reduce it,
    # and unfused indexer/scores, repeated loads and temporary writes may increase it.
    concat = 2 * (c['raw'] + c['csa'] + c['hca'])
    attention = 2 * (c['raw'] + c['hca'] + 21 * 512 * 512 * 2)
    return {'weight_proxy': WEIGHT_DECODE_BYTES, 'concat_proxy': concat,
            'lid_scan': c['lid'], 'attention_proxy': attention,
            'total': WEIGHT_DECODE_BYTES + concat + c['lid'] + attention}


def prefill_work(context, ubatch=512):
    # 13B matmul parameters/input token is an architecture-rounded proxy. Output
    # logits are needed at the final prompt position, not at every input token.
    matrix = 2 * 13e9 * context
    # Integral of causal prefix lengths: integral_0^N t dt = N^2/2.
    lid = 21 * 2 * 64 * 128 * context**2 / (2 * 4)
    hca = 20 * 4 * 64 * 512 * context**2 / (2 * 128)
    swa = 43 * 4 * 64 * 512 * 128 * context
    sparse_csa = 21 * 4 * 64 * 512 * 512 * context
    dense_csa = 21 * 4 * 64 * 512 * context**2 / (2 * 4)
    occupancy = 1 - (1 - 6 / 256)**ubatch
    chunks = math.ceil(context / ubatch)
    batch_weights = MODEL_BYTES - EXPERT_BYTES + occupancy * EXPERT_BYTES
    return {'matrix_FLOP': matrix, 'lid_FLOP': lid, 'hca_FLOP': hca, 'swa_FLOP': swa,
            'csa_ideal_selected_FLOP': sparse_csa, 'csa_full_compressed_FLOP': dense_csa,
            'attention_ideal_selected_FLOP': lid + hca + swa + sparse_csa,
            'attention_full_compressed_FLOP': lid + hca + swa + dense_csa,
            'ubatch': ubatch, 'chunks': chunks, 'expert_occupancy_proxy': occupancy,
            'weight_read_proxy_bytes': chunks * batch_weights,
            'unfused_lid_score_one_tensor_bytes': 64 * math.ceil(context / 4) * ubatch * 4}


def estimate(context, device, scenario, ubatch=512):
    spec = DEVICES[device]
    memory_efficiency, compute_efficiency, overhead_multiplier = SCENARIOS[scenario]
    bandwidth = spec['bandwidth_GB_s'] * 1e9
    effective_bandwidth = bandwidth * memory_efficiency
    # A hypothetical effective end-to-end operation rate, not a claim all Q8/F16
    # and MXFP4 kernels share a native FP4 peak. Native dense FP4 is a scale only.
    effective_compute = spec['dense_fp4_reference_TF_s'] * 1e12 * compute_efficiency
    traffic = decode_bytes(context, ubatch)
    work = prefill_work(context, ubatch)
    weight_time = work['weight_read_proxy_bytes'] / effective_bandwidth
    matrix_time = work['matrix_FLOP'] / effective_compute
    matmul_stage = max(weight_time, matrix_time)
    def prefill(attention_flops):
        # Serial matrix/attention stages; matrix compute and weight reads overlap.
        # Excludes tokenizer, HTTP, network/RPC, launches, top-k and extra KV traffic.
        return matmul_stage + attention_flops / effective_compute
    return {'context': context, 'device': device, 'scenario': scenario,
            'memory_efficiency_assumed': memory_efficiency,
            'effective_compute_TF_s_assumed': effective_compute / 1e12,
            'other_decode_s_assumed': spec['base_other_decode_s'] * overhead_multiplier,
            'weights_plus_cache_GiB': (MODEL_BYTES + sum(cache(context, ubatch).values())) / 2**30,
            'cache_and_state_GiB': sum(cache(context, ubatch).values()) / 2**30,
            'decode_traffic': traffic, 'prefill_work': work,
            'decode_bandwidth_only_reference_tps': bandwidth / traffic['total'],
            'decode_scenario_tps': 1 / (traffic['total'] / effective_bandwidth +
                                       spec['base_other_decode_s'] * overhead_multiplier),
            'prefill_weight_s': weight_time, 'prefill_matrix_s': matrix_time,
            'prefill_ideal_selected_s': prefill(work['attention_ideal_selected_FLOP']),
            'prefill_full_compressed_s': prefill(work['attention_full_compressed_FLOP'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        assert math.isclose(EXPERT_BYTES / 1e9, 147.169738752)
        assert math.isclose(sum(cache(1048576).values()) / 2**30, 6.740615234375)
        assert prefill_work(1_000_000)['lid_FLOP'] == 43008 * 1_000_000**2
        assert prefill_work(1_000_000)['hca_FLOP'] == 10240 * 1_000_000**2
        a = estimate(200_000, 'two_spark_layer_split', 'middle')
        b = estimate(1_000_000, 'two_spark_layer_split', 'middle')
        assert a['decode_scenario_tps'] > b['decode_scenario_tps']
        assert a['prefill_ideal_selected_s'] < b['prefill_ideal_selected_s']
        assert b['prefill_full_compressed_s'] > b['prefill_ideal_selected_s']
        print('Cache, FLOP, serial-stage and monotonicity self-checks passed')
        return
    result = {'status': 'uncalibrated scenarios, not measured forecasts or absolute bounds',
              'backend_commit': 'd81235049384534c167caea52b85a694f6103d14',
              'devices': DEVICES, 'scenarios': SCENARIOS,
              'assumptions': ['Actual occupied input tokens; model already loaded; no prefix reuse or speculation.',
                              'Single request, sequential layer split across two devices; no bandwidth doubling.',
                              'All experts, attention, KV on accelerators; source baseline CPU-MoE flags removed.',
                              'Efficiency, operation rate, extra latency and balanced routing are assumed, not fitted.',
                              'Prefill time is a resource model; overhead and extra cache traffic can make it worse.',
                              'Sparse/full CSA alternatives are algorithm scenarios, not guaranteed bounds.'],
              'rows': [estimate(n, device, scenario) for n in (200_000, 1_000_000)
                       for device in DEVICES for scenario in SCENARIOS],
              'large_batch_sensitivity': [estimate(n, device, 'middle', 2048)
                                          for n in (200_000, 1_000_000) for device in DEVICES]}
    result['middle_comparison'] = []
    for context in (200_000, 1_000_000):
        spark, pro = [r for r in result['rows'] if r['context'] == context and r['scenario'] == 'middle']
        price_ratio = DEVICES['two_pro6000_layer_split']['budget_cny'] / DEVICES['two_spark_layer_split']['budget_cny']
        speed_ratio = pro['decode_scenario_tps'] / spark['decode_scenario_tps']
        result['middle_comparison'].append({
            'context': context, 'budget_ratio_assumed': price_ratio,
            'decode_speed_ratio_in_scenario': speed_ratio,
            'capex_per_decode_tps_ratio_in_scenario': price_ratio / speed_ratio,
            'prefill_ideal_selected_speed_ratio_in_scenario': spark['prefill_ideal_selected_s'] / pro['prefill_ideal_selected_s'],
            'prefill_full_compressed_speed_ratio_in_scenario': spark['prefill_full_compressed_s'] / pro['prefill_full_compressed_s'],
            'warning': 'Derived from shared efficiency assumptions; not independent evidence of value.'})
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for r in result['rows']:
        print(r['context'], r['device'], r['scenario'],
              'decode', round(r['decode_scenario_tps'], 2),
              'prefill-minutes', round(r['prefill_ideal_selected_s']/60, 2),
              round(r['prefill_full_compressed_s']/60, 2))


if __name__ == '__main__':
    main()
