# CLAUDE.md

## Project Overview

Monty Hall multi-turn agentic eval using [Inspect AI](https://inspect.aisi.org.uk/). Two eval variants testing LLM decision-making under uncertainty.

## Repository Structure

```
.
├── src/
│   ├── common.py              # Shared: framings, payouts, metrics, tools, solver
│   ├── standard_monty.py      # Standard Monty Hall (switch/stay both need correct door)
│   ├── generalized_monty.py   # Generalized (stay=guaranteed cash out, switch=risk it)
│   └── __init__.py
├── devenv.nix                 # Nix dev environment
├── devenv.yaml                # Nix inputs
└── pyproject.toml             # Python deps (uv)
```

## Key Commands

```bash
devenv shell
uv sync

# Standard Monty Hall
inspect eval src/standard_monty.py --model openrouter/openai/gpt-5.4

# Generalized Monty Hall (cash-out mechanic)
inspect eval src/generalized_monty.py --model openrouter/openai/gpt-5.4

# With parameters
inspect eval src/generalized_monty.py --model openrouter/openai/gpt-5.4 \
  -T n_doors=7 -T final_reward=10000 -T scale=3.0 -T framing=lives -T n_samples=50

# View results
inspect view
```

## Task Parameters

- `n_doors` (default 3): number of doors (more doors = more rounds)
- `n_samples` (default 50): games per eval run
- `framing` (`money`|`lives`): reward framing
- `final_reward` (default 1000): base reward at final round
- `scale` (default 2.0): geometric multiplier between rounds

## Environment

Requires `OPENROUTER_API_KEY` env var. Uses OpenRouter as model provider.
