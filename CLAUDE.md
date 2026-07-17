# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal knowledge base for Deep Learning, RL, and Generative AI — written notes (`docs/`), paper case studies, and working code implementations (`code/`). The guiding principle is **no copy-paste**: everything should be understood and re-derived from first principles.

## Repository layout

- `docs/` — structured notes by chapter/topic, math derivations, and paper case studies
- `code/diffusion/` — self-contained Python package for the DDPM-on-MNIST implementation
- `scripts/` — Node.js tooling for rendering Markdown+LaTeX to PDF
- `output/pdf/` — rendered PDFs (generated artifacts, not source)

## Diffusion code (`code/diffusion/`)

**Environment (managed with `uv`):**
```sh
cd code/diffusion
uv sync          # install deps into .venv
uv run python train_mnist_ddpm.py
```

**Run tests:**
```sh
cd code/diffusion
uv run python -m pytest test_train_mnist_ddpm.py -v
# or run unittest directly:
uv run python test_train_mnist_ddpm.py
```

The test file imports `train_mnist_ddpm` as a module, so tests must be run from the `code/diffusion/` directory.

**Key files:**
- `train_mnist_ddpm.py` — canonical DDPM training script (noise schedule, UNet, EMA, sampling)
- `train_mnist_ddpm_ema.py` — variant with EMA weights
- `test_train_mnist_ddpm.py` — unit tests covering reproducibility, UNet/ResBlock/GroupNorm structure, EMA, and sampling

**Architecture:** Minimal UNet with sinusoidal time embeddings. The UNet uses `ResBlock` (with GroupNorm) in the current implementation. `NoiseSchedule` exposes `beta`, `alpha_bar`, and `posterior_variance`. `EMA` shadows model weights separately from the optimizer. `sample()` accepts `variance_type="beta"` or `"posterior"`.

**Device detection order:** MPS → CUDA → CPU (set automatically via `DEVICE`).

## PDF rendering (`scripts/`)

Renders Markdown files with LaTeX math (`$...$` and `$$...$$`) to PDF using KaTeX + Puppeteer. Requires a Chromium browser installed on the system (Edge, Chrome, or Chromium).

```sh
# Render a specific markdown file:
bash scripts/render_ddpm_derivation_pdf.sh [input.md] [output.pdf]

# Or directly:
node scripts/render_pdf.js docs/case-studies/module1-ddpm-math-derivation.md output/pdf/out.pdf
```

`npm install` inside `scripts/` is run automatically by the shell script if `node_modules` is absent.
