# CLAUDE.md

## Project Overview

Research project template with PyTorch + CUDA, vLLM, Modal, and LaTeX support. NixOS-based dev environment via devenv.

## Repository Structure

```
.
├── config/                # YAML configuration (add as needed)
├── src/                   # Source code
│   ├── main.py            #   Entry point — verifies PyTorch/CUDA
│   └── __main__.py        #   Enables: python -m src
├── report/                # LaTeX paper
│   ├── main.tex
│   └── references.bib
├── scripts/               # One-off analysis scripts
├── main_modal.py          # Modal cloud GPU template
├── devenv.nix             # Nix dev environment (CUDA, uv, texlive)
├── devenv.yaml            # Nix inputs
└── pyproject.toml         # Python deps (uv)
```

## Key Commands

All commands assume `devenv shell` (or prefix with `devenv shell --`):

```bash
# Enter dev shell (provides CUDA drivers, uv, texlive, PYTHONPATH=.)
devenv shell

# Install Python deps
uv sync                       # Core (torch, numpy)
uv sync --extra model-eval    # + vllm
uv sync --extra modal         # + modal
uv sync --extra all           # Everything

# Run locally
.venv/bin/python -m src

# Run on Modal (cloud GPU)
.venv/bin/modal run main_modal.py

# Compile LaTeX report
cd report && latexmk -pdf main.tex
```

## Environment Setup

Uses [devenv](https://devenv.sh/) for reproducible dev environments on NixOS. Provides CUDA driver access, `uv`, `texliveFull`, and sets `PYTHONPATH=.`.

```bash
devenv shell
uv sync --extra all
```

For Modal: `modal setup` then `modal secret create huggingface-secret HF_TOKEN=your_token`
