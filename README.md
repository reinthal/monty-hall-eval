[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

# Research Template

Template repo. Clone. Research. Publish. No configure.

## What Inside

- **PyTorch + CUDA** — GPU go brrr. NixOS handle driver pain
- **vLLM** — serve big language model. Fast
- **Modal** — cloud GPU when local GPU not enough
- **LaTeX** — write paper like scientist
- **devenv** — one `devenv shell`, all tool ready. No apt. No brew. Reproducible
- **uv** — fast Python package. Replace pip. No wait
- **SOPS** — secret stay secret. Encrypt config, commit safe

## CI/CD

Push to main. Magic happen:

1. **Commitizen** bump version from commit message (`feat:`, `fix:`, etc.)
2. **CHANGELOG** update automatic
3. **GitHub Release** create with changelog body
4. **LaTeX PDF** compile and attach to release

No PAT needed. `GITHUB_TOKEN` do all work.

## Quick Start

```bash
# Enter cave (dev shell)
devenv shell

# Install dependency
uv sync --extra all

# Run code
.venv/bin/python -m src

# Run on cloud GPU
.venv/bin/modal run main_modal.py

# Make paper
cd report && latexmk -pdf main.tex
```

## Use as Template

1. Click **"Use this template"** on GitHub
2. Clone new repo
3. `devenv shell && uv sync`
4. Research

All workflow file copy over. Secret (`GITHUB_TOKEN`) auto-provided. No setup needed.

## Commit Convention

Use conventional commit. Caveman remember:

| Prefix | What do |
|--------|---------|
| `feat:` | New thing. Minor bump |
| `fix:` | Bug dead. Patch bump |
| `docs:` | Word change only |
| `refactor:` | Same behavior, better code |
| `BREAKING CHANGE:` | Big change. Major bump |

## Project Structure

```
.
├── src/               # Brain of operation
│   ├── main.py        #   Entry point
│   └── __main__.py    #   python -m src
├── report/            # LaTeX paper
│   ├── main.tex
│   └── references.bib
├── config/            # YAML config go here
├── scripts/           # One-off script
├── main_modal.py      # Modal cloud GPU template
├── devenv.nix         # Nix environment (CUDA, uv, texlive)
├── pyproject.toml     # Python deps + commitizen config
└── .github/workflows/ # CI/CD automation
```
