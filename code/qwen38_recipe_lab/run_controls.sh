#!/bin/zsh
# Serial, predeclared matched-token controls; never overwrite an existing run.
set -e
cd "${0:A:h}"
uv run python train.py --size mac --data data/shakespeare --out runs/qwen-control-muon --device mps --steps 256 --seq 128 --accum 1 --eval-every 100 --lr .0003 --schedule constant --optimizer muon --gr-mode gated > runs/qwen-control-muon.log 2>&1
uv run python train.py --size mac --data data/shakespeare --out runs/qwen-control-adamw --device mps --steps 256 --seq 128 --accum 1 --eval-every 100 --lr .0003 --schedule constant --optimizer adamw --gr-mode gated > runs/qwen-control-adamw.log 2>&1
uv run python train.py --size mac --data data/shakespeare --out runs/qwen-control-ungated --device mps --steps 256 --seq 128 --accum 1 --eval-every 100 --lr .0003 --schedule constant --optimizer muon --gr-mode ungated > runs/qwen-control-ungated.log 2>&1
uv run python train.py --stage pretrain --init runs/qwen-cpt/last.pt --data data/owt --mix-data data/shakespeare --out runs/qwen-dense-control --device mps --steps 200 --seq 128 --accum 1 --eval-every 100 --lr .000025 --schedule constant > runs/qwen-dense-control.log 2>&1
