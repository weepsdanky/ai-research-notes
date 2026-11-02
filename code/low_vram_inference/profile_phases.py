"""Bounded Windows CPU/page-fault or CUDA trace of one existing GGUF request.

Uses installed Windows Performance Toolkit/Nsight only. Raw system ETL stays on the Windows host;
export only the target process when sharing evidence.
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from run_deepseek import monitor, request

NSYS = Path(r'D:\nsys26\ProgramFiles64Folder\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe')
XPERF = Path(r'C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe')
PROMPT = 'Write a detailed English explanation of how a computer reads data from an SSD. Continue for at least 300 words.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['control', 'etw', 'cuda'], required=True)
    parser.add_argument('--root', type=Path, default=Path(r'D:\ai-research\low-vram-inference'))
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--nsys', type=Path, default=NSYS)
    parser.add_argument('--cuda-capture-seconds', type=int, default=60)
    parser.add_argument('--kmp-blocktime', choices=['unset', '0'])
    args = parser.parse_args()
    nsys = args.nsys
    if os.name != 'nt' or not 16 <= args.tokens <= 128 or not 20 <= args.cuda_capture_seconds <= 120:
        raise RuntimeError('Windows only; output budget must be 16..128 tokens')
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError('An elevated session is required for ETW')
    if min(shutil.disk_usage(drive).free for drive in ('C:\\', 'D:\\')) < 40e9:
        raise RuntimeError('Keep at least 40GB free on both trace/model drives')
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    base_url = 'http://127.0.0.1:18080'
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 18080)) == 0:
            raise RuntimeError('Port 18080 is already occupied')
    if args.mode == 'etw':
        status = subprocess.run(['wpr', '-status'], capture_output=True, text=True)
        # The host uses English WPR output; never cancel another recording.
        if 'WPR is not recording' not in status.stdout:
            raise RuntimeError('WPR is already recording or status is unknown: ' + status.stdout)
        loggers = subprocess.run([str(XPERF), '-loggers'], capture_output=True, text=True)
        if loggers.returncode or 'NT Kernel Logger' in loggers.stdout:
            raise RuntimeError('Kernel logger is occupied or status is unknown')
    if args.mode == 'cuda' and not nsys.is_file():
        raise FileNotFoundError(nsys)
    model = args.root / 'models/deepseek-q4/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf'
    from download_deepseek import require_verified
    manifest = json.loads(Path(__file__).with_name('deepseek-q4-manifest.json').read_text())
    require_verified(model.parent, manifest)
    folder = args.root / 'results' / ('phase-' + args.mode + '-' + time.strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)
    session = 'LowVram' + str(os.getpid())
    server_command = [str(args.root / 'bin/llama-server.exe'), '-m', str(model),
                      '--load-mode', 'mmap', '--no-repack', '--no-host', '--lazy-mode', 'on',
                      '--cpu-moe', '--no-op-offload', '--no-kv-offload', '-ngl', '32',
                      '-c', '512', '-b', '32', '-ub', '32', '-t', '16', '-np', '1',
                      '--jinja', '--reasoning', 'off', '--host', '127.0.0.1', '--port', '18080', '-v']
    command = server_command
    child_environment = os.environ.copy()
    if args.kmp_blocktime == 'unset':
        child_environment.pop('KMP_BLOCKTIME', None)
    elif args.kmp_blocktime == '0':
        child_environment['KMP_BLOCKTIME'] = '0'
    if args.mode == 'cuda':
        command = [str(nsys), 'launch', '--session-new=' + session, '--trace=cuda',
                   '--cuda-graph-trace=graph:host-only', '--cuda-flush-interval=1000', '--show-output=true'] + command
    result = {'mode': args.mode, 'command': command, 'server_command': server_command,
              'cache_note': 'Fresh process; OS file cache uncontrolled; identical short prompt. Profilers run separately.',
              'started_unix_s': time.time(), 'request_budget_s': 360,
              'kmp_blocktime_override': args.kmp_blocktime,
              'logical_processors': os.cpu_count(),
              'openmp_environment': {name: child_environment.get(name) for name in (
                  'OMP_WAIT_POLICY', 'KMP_BLOCKTIME', 'KMP_USE_YIELD', 'OMP_NUM_THREADS', 'OMP_PROC_BIND', 'KMP_AFFINITY')}}
    (folder / 'command.json').write_text(json.dumps(result, indent=2))
    print(str(folder), flush=True)
    process = None
    launcher = None
    done = threading.Event()
    worker = None
    timer = None
    trace_timer = None
    trace_stop_errors = []
    recording = False

    def tool(command):
        began = time.time()
        call = subprocess.run(command, capture_output=True, text=True, errors='replace',
                              timeout=600 if '-d' in command or command[1] in ('stop', 'shutdown') else 120)
        with (folder / 'profiler-commands.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'command': command, 'start_unix_s': began,
                                     'end_unix_s': time.time(), 'returncode': call.returncode,
                                     'stdout': call.stdout, 'stderr': call.stderr}) + '\n')
        if call.returncode:
            raise RuntimeError(str(command[:3]) + ': ' + call.stdout + call.stderr)
        return call

    def stop_cuda():
        nonlocal recording
        result['cuda_stop_requested_unix_s'] = time.time()
        try:
            tool([str(nsys), 'stop', '--session=' + session])
            recording = False
        except Exception as error:
            trace_stop_errors.append(repr(error))
        result['cuda_stop_finished_unix_s'] = time.time()

    with (folder / 'server.log').open('wb') as log:
        try:
            launcher = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        env=child_environment)
            deadline = time.monotonic() + 180
            while True:
                if launcher.poll() is not None and (args.mode != 'cuda' or launcher.returncode != 0):
                    raise RuntimeError('Launcher exited: ' + str(launcher.returncode))
                if time.monotonic() > deadline:
                    raise TimeoutError('Server readiness exceeded 180s')
                try:
                    with urllib.request.urlopen(base_url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            if args.mode == 'cuda':
                # Nsight owns a child process. Resolve exactly our isolated listening port.
                pid_text = tool(['powershell', '-NoProfile', '-Command',
                                 '(Get-NetTCPConnection -State Listen -LocalPort 18080).OwningProcess']).stdout
                target_pid = int(pid_text.strip())
                from ctypes import wintypes as w
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
                kernel.OpenProcess.restype = w.HANDLE
                kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
                kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
                kernel.CloseHandle.argtypes = [w.HANDLE]
                class Target:
                    pid = target_pid
                    _handle = kernel.OpenProcess(0x101411, False, target_pid)
                    def poll(self):
                        code = w.DWORD()
                        kernel.GetExitCodeProcess(self._handle, ctypes.byref(code))
                        return None if code.value == 259 else code.value
                    def terminate(self):
                        kernel.TerminateProcess(self._handle, 1)
                process = Target()
                if not process._handle:
                    raise ctypes.WinError()
            else:
                process = launcher
            result['server_pid'] = process.pid
            (folder / 'server.pid').write_text(str(process.pid))
            if args.mode == 'cuda':
                modules = tool(['powershell', '-NoProfile', '-Command',
                                '(Get-Process -Id ' + str(process.pid) + ').Modules | '
                                "Select-Object ModuleName,@{n='base';e={$_.BaseAddress.ToInt64()}},ModuleMemorySize | ConvertTo-Json"])
                (folder / 'module-map.json').write_text(modules.stdout, encoding='utf-8')
            worker = threading.Thread(target=monitor, args=(process, done, folder / 'samples.jsonl'))
            worker.start()
            time.sleep(2)
            if args.mode == 'etw':
                tool([str(XPERF), '-on',
                      'PROC_THREAD+LOADER+CSWITCH+DISPATCHER+PROFILE+HARD_FAULTS+DISK_IO+DISK_IO_INIT',
                      '-BufferSize', '1024', '-MinBuffers', '32', '-MaxBuffers', '128',
                      '-f', str(folder / 'kernel-raw.etl')])
                recording = True
                tool([str(XPERF), '-m', 'LowVramRequestStart'])
            elif args.mode == 'cuda':
                tool([str(nsys), 'start', '--session=' + session, '--sample=none', '--cpuctxsw=none',
                      '--export=sqlite', '--output=' + str(folder / 'cuda')])
                recording = True
                trace_timer = threading.Timer(args.cuda_capture_seconds, stop_cuda)
                trace_timer.start()
            timer = threading.Timer(360, process.terminate)
            timer.start()
            result['request_started_unix_s'] = time.time()
            entry = request(PROMPT, folder / 'request-0.jsonl', base_url, max_tokens=args.tokens)
            result['request_ended_unix_s'] = time.time()
            result['request'] = entry
            (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            if args.mode == 'etw':
                tool([str(XPERF), '-m', 'LowVramRequestEnd'])
            count = (entry.get('usage') or {}).get('completion_tokens', 0)
            if count != args.tokens or not entry['text'].strip():
                raise RuntimeError('Expected full output token budget and nonempty text')
            if args.mode == 'control':
                # Bracket the request end for cumulative CPU-counter interpolation.
                time.sleep(2)
            result['status'] = 'request-passed'
        except Exception as error:
            result['status'], result['error'] = 'failed', repr(error)
        finally:
            if timer:
                timer.cancel()
            if trace_timer:
                trace_timer.cancel()
                trace_timer.join(timeout=620)
                if trace_timer.is_alive() or trace_stop_errors:
                    result['status'] = 'failed'
                    result['cuda_stop_errors'] = trace_stop_errors
            if args.mode == 'etw' and process and process.poll() is None:
                # Close the 155GB mapping before the final process/image rundown.
                process.terminate()
                launcher.wait(timeout=20)
            if recording:
                try:
                    if args.mode == 'etw':
                        tool([str(XPERF), '-d', str(folder / 'cpu-faults.etl')])
                    else:
                        tool([str(nsys), 'stop', '--session=' + session])
                except Exception as error:
                    result['profiler_stop_error'] = repr(error)
                    result['status'] = 'failed'
            done.set()
            if process and process.poll() is None:
                process.terminate()
            if worker:
                worker.join(timeout=15)
            if args.mode == 'cuda' and launcher:
                try:
                    tool([str(nsys), 'shutdown', '--session=' + session, '--kill=true'])
                except Exception as error:
                    result['profiler_shutdown_error'] = repr(error)
                if process:
                    kernel.CloseHandle(process._handle)
            if launcher:
                try:
                    launcher.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    launcher.terminate()
                    launcher.wait(timeout=10)
            if (folder / 'samples.error.txt').exists() or (worker and worker.is_alive()):
                result['status'] = 'failed'
                result['monitoring_error'] = 'Inspect samples.error.txt / monitor did not finish'
            (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result['status'], flush=True)
    if result['status'] != 'request-passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
