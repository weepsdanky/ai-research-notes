"""Windows native server experiment; stdlib only, bounded, saves raw evidence."""
import argparse
import ctypes
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    'What is 2 + 3? Answer with only the number.',
    'Translate into Chinese, with no explanation: The cat is sleeping.',
]


def content_from_event(event):
    choices = event.get('choices', [])
    return choices[0].get('delta', {}).get('content', '') if choices else ''


def content_passes(prompt, text):
    if prompt == PROMPTS[0]:
        return text.strip() == '5'
    return '猫' in text and '睡' in text and not any(word in text for word in ('没有', '不', '沒'))


def monitor(process, done, output):
    """Process RAM and PDH GPU usage; retain machine disk and device GPU counters."""
    from ctypes import wintypes as w
    class Memory(ctypes.Structure):
        _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged',
            'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    class Value(ctypes.Structure):
        _fields_ = [('status', w.DWORD), ('value', ctypes.c_double)]
    class GpuValue(ctypes.Structure):
        _fields_ = [('status', w.DWORD), ('value', ctypes.c_longlong)]
    class GpuItem(ctypes.Structure):
        _fields_ = [('name', w.LPWSTR), ('value', GpuValue)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    kernel.GetProcessIoCounters.argtypes = [w.HANDLE, ctypes.POINTER(IoCounters)]
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    pdh = ctypes.WinDLL('pdh')
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Memory), w.DWORD]
    pdh.PdhOpenQueryW.argtypes = [w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
    pdh.PdhAddEnglishCounterW.argtypes = [w.HANDLE, w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(w.HANDLE)]
    pdh.PdhCollectQueryData.argtypes = [w.HANDLE]
    pdh.PdhGetFormattedCounterValue.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.POINTER(Value)]
    pdh.PdhCloseQuery.argtypes = [w.HANDLE]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.POINTER(w.DWORD), ctypes.c_void_p]
    pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long

    def gpu_usage(handle, add_status):
        if add_status or not handle:
            return {'value': None, 'status': add_status or 'missing_handle', 'instances': []}
        for _ in range(3):
            size, count = w.DWORD(), w.DWORD()
            status = pdh.PdhGetFormattedCounterArrayW(handle, 0x1400, ctypes.byref(size), ctypes.byref(count), None) & 0xffffffff
            if status not in (0, 0x800007d2) or not size.value:
                return {'value': None, 'status': status, 'instances': []}
            buffer = ctypes.create_string_buffer(size.value)
            status = pdh.PdhGetFormattedCounterArrayW(handle, 0x1400, ctypes.byref(size), ctypes.byref(count), buffer) & 0xffffffff
            if status == 0x800007d2:
                continue
            if status:
                return {'value': None, 'status': status, 'instances': []}
            items = ctypes.cast(buffer, ctypes.POINTER(GpuItem))
            matched = {}
            for i in range(count.value):
                item = items[i]
                if (item.name or '').startswith('pid_' + str(process.pid) + '_'):
                    value = item.value.value if item.value.status in (0, 1) and item.value.value >= 0 else None
                    matched[item.name] = {'name': item.name, 'value': value, 'status': item.value.status}
            entries = list(matched.values())
            total = sum(x['value'] for x in entries) if entries and all(x['value'] is not None for x in entries) else None
            return {'value': total, 'status': 0 if entries else 'no_matching_pid', 'instances': entries}
        return {'value': None, 'status': 'instances_changed_during_read', 'instances': []}

    query, counter = w.HANDLE(), w.HANDLE()
    try:
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)):
            raise RuntimeError('PdhOpenQuery failed')
        if pdh.PdhAddEnglishCounterW(query, r'\PhysicalDisk(_Total)\Disk Read Bytes/sec', 0, ctypes.byref(counter)):
            raise RuntimeError('PhysicalDisk read counter unavailable')
        gpu_counters = {}
        for kind, name in (('dedicated', 'Dedicated Usage'), ('shared', 'Shared Usage')):
            gpu_counter = w.HANDLE()
            add_status = pdh.PdhAddEnglishCounterW(query, '\\GPU Process Memory(*)\\' + name, 0, ctypes.byref(gpu_counter))
            gpu_counters[kind] = (gpu_counter, add_status)
        pdh.PdhCollectQueryData(query)
        started = time.perf_counter()
        with output.open('w', encoding='utf-8') as stream:
            while not done.is_set():
                sample = {'elapsed_s': time.perf_counter() - started, 'unix_s': time.time()}
                mem = Memory(cb=ctypes.sizeof(Memory))
                if psapi.GetProcessMemoryInfo(w.HANDLE(int(process._handle)), ctypes.byref(mem), mem.cb):
                    sample.update(rss_bytes=mem.rss, peak_rss_bytes=mem.peak_rss,
                                  private_bytes=mem.private, page_fault_count=mem.faults)
                elif process.poll() is None:
                    raise ctypes.WinError(ctypes.get_last_error())
                times = [w.FILETIME() for _ in range(4)]
                if kernel.GetProcessTimes(w.HANDLE(int(process._handle)), *[ctypes.byref(t) for t in times]):
                    for name, t in zip(('kernel', 'user'), times[2:]):
                        sample['process_cpu_' + name + '_seconds'] = ((t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e7
                io = IoCounters()
                if kernel.GetProcessIoCounters(w.HANDLE(int(process._handle)), ctypes.byref(io)):
                    sample['process_io_read_bytes'] = io.read_bytes
                    sample['process_io_read_operations'] = io.read_ops
                pdh.PdhCollectQueryData(query)
                value = Value()
                status = pdh.PdhGetFormattedCounterValue(counter, 0x1200, None, ctypes.byref(value))
                sample['machine_disk_read_bytes_per_s'] = value.value if status == 0 and value.status in (0, 1) else None
                sample['gpu_process_instances'] = {}
                for kind, handles in gpu_counters.items():
                    info = gpu_usage(*handles)
                    sample['gpu_process_instances'][kind] = info
                    sample['process_gpu_' + kind + '_usage_bytes'] = info['value']
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10)
                if gpu.returncode:
                    raise RuntimeError('nvidia-smi failed: ' + gpu.stderr)
                sample['device_used_vram_mib'] = int(gpu.stdout.strip().splitlines()[0])
                stream.write(json.dumps(sample) + '\n')
                stream.flush()
                done.wait(1)
    except Exception as error:
        output.with_suffix('.error.txt').write_text(repr(error), encoding='utf-8')
    finally:
        if query:
            pdh.PdhCloseQuery(query)


def request(prompt, output, base_url, max_tokens=64):
    payload = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
               'reasoning_effort': 'none', 'max_tokens': max_tokens, 'stream': True,
               'cache_prompt': False, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(base_url + '/v1/chat/completions',
                                 json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    began, first_content = time.perf_counter(), None
    events, text, usage = [], '', None
    with urllib.request.urlopen(req, timeout=3600) as response, output.open('w', encoding='utf-8') as stream:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:])
            delta = content_from_event(event)
            if delta:
                if first_content is None:
                    first_content = time.perf_counter() - began
                text += delta
            if event.get('usage'):
                usage = event['usage']
            record = {'elapsed_s': time.perf_counter() - began, 'event': event}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            events.append(record)
    return {'prompt': prompt, 'text': text, 'time_to_first_content_s': first_content,
            'total_s': time.perf_counter() - began, 'usage': usage,
            'timings': [e['event']['timings'] for e in events if 'timings' in e['event']]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(r'D:\ai-research\low-vram-inference'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workload', choices=['fixed', 'limits', 'profile'], default='fixed')
    parser.add_argument('--label', default='deepseek')
    parser.add_argument('--port', type=int, default=18080)
    args = parser.parse_args()
    base_url = "http://127.0.0.1:" + str(args.port)
    # Local experiments must bypass any system HTTP proxy.
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    if os.name != 'nt':
        raise RuntimeError('This experiment uses Windows native counters')
    model = args.model or args.root / 'models/deepseek-q4/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf'
    if args.model is None:
        manifest = json.loads(Path(__file__).with_name('deepseek-q4-manifest.json').read_text())
        for item in manifest['files']:
            shard = model.parent / Path(item['path']).name
            if not shard.is_file() or shard.stat().st_size != item['size']:
                raise RuntimeError('Target shard missing/incomplete: ' + str(shard))
        from download_deepseek import require_verified
        require_verified(model.parent, manifest)
    if not model.is_file():
        raise FileNotFoundError(model)
    # No automatic parallel experiment: refuse to collide with an existing local service.
    try:
        urllib.request.urlopen(base_url + '/health', timeout=2)
    except urllib.error.URLError as error:
        if isinstance(error, urllib.error.HTTPError):
            raise RuntimeError('Experiment port already has an HTTP service') from error
    else:
        raise RuntimeError('Experiment port already has an HTTP service')
    folder = args.root / 'results' / (args.label + '-ngl' + str(args.gpu_layers) + '-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)
    command = [str(args.root / 'bin/llama-server.exe'), '-m', str(model),
               '--load-mode', 'mmap', '--no-repack', '--no-host', '--lazy-mode', 'on',
               '--cpu-moe', '--no-op-offload', '--no-kv-offload', '-ngl', str(args.gpu_layers),
               '-c', '512', '-b', '32', '-ub', '32', '-t', str(args.threads), '-np', '1',
               '--jinja', '--reasoning', 'off', '--host', '127.0.0.1', '--port', str(args.port), '-v']
    result = {'command': command, 'cache_note': 'Fresh process; OS file cache uncontrolled. Same-server repeated requests are warm.'}
    (folder / 'command.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    done = threading.Event()
    started = time.perf_counter()
    with (folder / 'server.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        (folder / 'server.pid').write_text(str(process.pid))
        worker = threading.Thread(target=monitor, args=(process, done, folder / 'samples.jsonl'))
        worker.start()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server exited: ' + str(process.returncode))
                if time.perf_counter() - started > 1800:
                    raise TimeoutError('Server not ready within 1800 seconds')
                try:
                    with urllib.request.urlopen(base_url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except urllib.error.URLError:
                    pass
                time.sleep(1)
            result['process_start_to_ready_s'] = time.perf_counter() - started
            time.sleep(2)  # Obtain two valid counter intervals even for tiny smoke models.
            result['requests'] = []
            prompts = PROMPTS
            if args.workload != 'fixed':
                short = 'Write a detailed English explanation of how a computer reads data from an SSD. Continue for at least 300 words.'
                prompts = [short, 'Context: ' + 'Storage devices contain data. ' * 24 + '\n' + short]
            for run, prompt in enumerate(prompts + prompts * (0 if args.workload == 'profile' else 2 if args.workload == 'limits' else 3)):
                request_started = time.time()
                entry = request(prompt, folder / ('request-' + str(run) + '.jsonl'), base_url, max_tokens=128 if args.workload != 'fixed' else 64)
                entry['request_started_unix_s'] = request_started
                entry['phase'] = 'first-pass' if run < len(prompts) else 'same-server-repeat'
                entry['repeat_index'] = 0 if run < len(prompts) else (run - len(prompts)) // len(prompts) + 1
                result['requests'].append(entry)
                (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            for entry in result['requests']:
                if not (entry.get('usage', {}).get('completion_tokens', 0) >= 64 and bool(entry['text'].strip()) if args.workload != 'fixed' else content_passes(entry['prompt'], entry['text'])):
                    raise RuntimeError('Content gate failed: ' + entry['phase'])
            if (folder / 'samples.error.txt').exists():
                raise RuntimeError('Monitoring failed; inspect samples.error.txt')
            result['status'] = 'content-and-monitoring-gates-passed'
        except Exception as error:
            result['status'], result['error'] = 'failed', repr(error)
        finally:
            done.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            worker.join(timeout=15)
            result['server_exit_code'] = process.returncode
            sample_path = folder / 'samples.jsonl'
            samples = [json.loads(line) for line in sample_path.read_text().splitlines()] if sample_path.exists() else []
            valid = [s for s in samples if 'rss_bytes' in s and s.get('machine_disk_read_bytes_per_s') is not None]
            if len(valid) < 2 or (folder / 'samples.error.txt').exists() or worker.is_alive():
                result['status'] = 'failed'
                result['monitoring_error'] = 'Need two valid resource samples and a cleanly completed monitor'
            else:
                result['resources'] = {
                    'peak_process_rss_bytes': max(s['peak_rss_bytes'] for s in valid),
                    'peak_sampled_private_bytes': max(s['private_bytes'] for s in valid),
                    'peak_sampled_device_used_vram_mib': max(s['device_used_vram_mib'] for s in valid),
                    'machine_disk_read_bytes_approx': sum(
                        b['machine_disk_read_bytes_per_s'] * (b['elapsed_s'] - a['elapsed_s'])
                        for a, b in zip(valid, valid[1:])),
                    'sample_count': len(valid),
                    'note': 'Disk is whole-machine estimated physical reads; VRAM is whole-device sampled usage. Page faults include soft faults.'}

                for kind in ('dedicated', 'shared'):
                    values = [s['process_gpu_' + kind + '_usage_bytes'] for s in valid if s.get('process_gpu_' + kind + '_usage_bytes') is not None]
                    result['resources']['peak_sampled_process_gpu_' + kind + '_usage_bytes'] = max(values) if values else None
                result['resources']['gpu_counter_note'] = 'PDH process GPU usage is a counter proxy; retain instances and cross-check device usage/CUDA buffers.'
            (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(folder, result['status'], flush=True)
    if result['status'] != 'content-and-monitoring-gates-passed':
        raise SystemExit(1)


if __name__ == '__main__':
    import sys
    if '--self-check' in sys.argv:
        assert content_from_event({'choices': [{'delta': {'content': '5'}}]}) == '5'
        assert content_from_event({'choices': []}) == ''
        assert content_from_event({'choices': [{'delta': {'reasoning_content': 'thinking'}}]}) == ''
        assert content_passes(PROMPTS[0], '5')
        assert not content_passes(PROMPTS[0], '4')
        assert content_passes(PROMPTS[1], '猫正在睡觉。')
        assert not content_passes(PROMPTS[1], '猫没有睡觉。')
        print('SSE extraction and output-gate self-check passed')
    else:
        main()
