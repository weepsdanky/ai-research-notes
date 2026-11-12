# Qwen3.8-Flash-Next Mac recipe lab

Source snapshot from local revision 495797b for the scaled Mac experiments.

Run from this directory with Python 3.11 and uv.

```sh
uv sync --python 3.11
uv run python bootstrap.py
uv run python -m unittest -v
```

Profiles and source locks are included. Checkpoints, corpora, predictions and experiment reports are stored separately.
