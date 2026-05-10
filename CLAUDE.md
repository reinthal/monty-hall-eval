# CLAUDE.md

## Project Overview

Monty Hall multi-turn agentic eval using [Inspect AI](https://inspect.aisi.org.uk/). Tests whether LLMs discover the optimal "always switch" strategy.

## Repository Structure

```
.
├── src/
│   ├── monty_hall.py      # Eval task definition (tools + scorer)
│   └── __init__.py
├── devenv.nix             # Nix dev environment
├── devenv.yaml            # Nix inputs
└── pyproject.toml         # Python deps (uv)
```

## Key Commands

```bash
devenv shell
uv sync

# Run eval against GPT-5.4 via OpenRouter
inspect eval src/monty_hall.py --model openrouter/openai/gpt-5.4

# Run eval against Opus 4.6 via OpenRouter
inspect eval src/monty_hall.py --model openrouter/anthropic/claude-opus-4-6

# View results
inspect view
```

## Environment

Requires `OPENROUTER_API_KEY` env var. Uses OpenRouter as model provider.
