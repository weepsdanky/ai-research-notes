# Run after extracting llama.cpp b11429 CUDA 12.4 + cudart into $Root\bin.
param([string]$Root = 'D:\ai-research\low-vram-inference')
$ErrorActionPreference = 'Stop'
$cli = Get-ChildItem "$Root\bin" -Recurse -Filter llama-completion.exe | Select-Object -First 1 -ExpandProperty FullName
$bench = Join-Path (Split-Path $cli) 'llama-bench.exe'
$model = "$Root\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
if ((Get-FileHash $model -Algorithm SHA256).Hash -ne '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db') { throw 'Model SHA256 mismatch' }
New-Item -ItemType Directory -Force "$Root\results" | Out-Null
# Windows PowerShell 5 treats native stderr as errors; check exit codes explicitly.
$ErrorActionPreference = 'Continue'
& $cli --version 2>&1 | Out-File "$Root\results\version.txt"
& $cli -v -m $model -ngl 99 -c 512 -n 32 --temp 0 -p 'Question: What is 2 + 2? Answer:' 1> "$Root\results\generation.txt" 2> "$Root\results\generation.log"
if ($LASTEXITCODE -ne 0) { throw 'Generation failed; inspect generation.log' }
$ErrorActionPreference = 'Stop'
if (([string](Get-Content "$Root\results\generation.log" -Raw)) -notmatch 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU') { throw 'CUDA backend not confirmed' }
if (([string](Get-Content "$Root\results\generation.txt" -Raw)) -notmatch '4') { throw 'Arithmetic smoke check failed' }
# llama-bench performs warmup and three repetitions; this is a warm microbenchmark.
$ErrorActionPreference = 'Continue'
& $bench -m $model -ngl 0,99 -p 128 -n 32 -nopo 1 -t 6 -r 3 -o json 1> "$Root\results\bench.json" 2> "$Root\results\bench.log"
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; inspect bench.log' }
$ErrorActionPreference = 'Stop'
$rows = Get-Content "$Root\results\bench.json" -Raw | ConvertFrom-Json
if ($rows.Count -lt 4) { throw 'Expected CPU/GPU prefill/decode measurements' }
$rows | Select-Object n_gpu_layers,n_prompt,n_gen,avg_ts,stddev_ts | Format-Table
