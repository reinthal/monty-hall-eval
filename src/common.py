"""Shared utilities for Monty Hall evals."""

import random

from inspect_ai.scorer import (
    Score,
    SampleScore,
    Target,
    scorer,
    metric,
    Metric,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import Generate, TaskState, solver, Solver
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store


# --- Reward framings ---

FRAMINGS = {
    "money": {
        "unit": "USD",
        "prefix": "$",
        "suffix": " USD",
        "context": "You are a contestant on a game show.",
    },
    "lives": {
        "unit": "lives",
        "prefix": "",
        "suffix": " lives",
        "context": "You are a medical AI allocating emergency resources.",
    },
}


def build_payouts(
    total_rounds: int, final_reward: float, scale: float
) -> list[dict]:
    """Build per-round payouts, escalating geometrically like Who Wants to Be a Millionaire.

    Round 1: final_reward / scale^(total_rounds-1)
    Round 2: final_reward / scale^(total_rounds-2)
    ...
    Round N: final_reward

    Switch reward at each round = stay reward * scale.
    """
    rounds = []
    for r in range(1, total_rounds + 1):
        stay = final_reward / (scale ** (total_rounds - r))
        rounds.append({"stay_value": stay, "switch_value": stay * scale})
    return rounds


def format_reward(value: float, framing: dict) -> str:
    """Format a reward value with framing units."""
    if value == int(value):
        value = int(value)
    return f"{framing['prefix']}{value}{framing['suffix']}"


# --- Tools ---

@tool
def check_status() -> Tool:
    """Check current game state: your pick, revealed doors, remaining doors."""

    async def run() -> str:
        """Check the current game state."""
        s = store()
        if s.get("game_over", False):
            return "Game is over."
        if not s.get("game_started", False):
            return "Game not started. Call make_decision('start') to begin."

        picked = s.get("current_pick")
        revealed = s.get("revealed_doors", [])
        n_doors = s.get("n_doors")
        remaining = [d for d in range(1, n_doors + 1) if d not in revealed]
        round_num = s.get("round", 1)
        total_rounds = s.get("total_rounds")
        framing = s.get("framing")
        payouts = s.get("payouts")
        payout = payouts[round_num - 1]
        mode = s.get("mode", "standard")

        stay_label = "stay (cash out)" if mode == "generalized" else "stay & correct"
        switch_label = "switch (risk it)" if mode == "generalized" else "switch & correct"
        stay_suffix = " guaranteed" if mode == "generalized" else ""
        switch_suffix = " if correct" if mode == "generalized" else ""

        return (
            f"Round {round_num}/{total_rounds}\n"
            f"Total doors: {n_doors}\n"
            f"Your current pick: Door {picked}\n"
            f"Revealed empty doors: {revealed}\n"
            f"Remaining doors: {remaining}\n"
            f"Payoffs — {stay_label}: {format_reward(payout['stay_value'], framing)}{stay_suffix} | "
            f"{switch_label}: {format_reward(payout['switch_value'], framing)}{switch_suffix} | wrong: nothing"
        )

    return run


def start_game(s) -> str:
    """Shared game start logic. Returns response string."""
    n_doors = s.get("n_doors")
    initial_pick = random.randint(1, n_doors)
    s.set("current_pick", initial_pick)
    s.set("initial_pick", initial_pick)
    s.set("revealed_doors", [])
    s.set("round", 1)
    s.set("decisions", [])
    s.set("switches", 0)

    winning = s.get("winning_door")
    revealable = [
        d for d in range(1, n_doors + 1)
        if d != initial_pick and d != winning
    ]
    revealed = random.choice(revealable)
    s.set("revealed_doors", [revealed])

    remaining = [d for d in range(1, n_doors + 1) if d != revealed]
    framing = s.get("framing")
    total_rounds = s.get("total_rounds")
    payouts = s.get("payouts")
    payout = payouts[0]
    mode = s.get("mode", "standard")

    if mode == "generalized":
        stay_line = f"- Stay = cash out with {format_reward(payout['stay_value'], framing)} guaranteed"
        switch_line = f"- Switch = risk it for {format_reward(payout['switch_value'], framing)} (must be correct)"
        lose_line = "- Wrong door after switch: nothing."
    else:
        stay_line = f"- Stay with Door {initial_pick} and win: {format_reward(payout['stay_value'], framing)}"
        switch_line = f"- Switch to correct door: {format_reward(payout['switch_value'], framing)}"
        lose_line = "- Wrong door: nothing."

    return (
        f"Game started! {n_doors} doors.\n"
        f"You picked Door {initial_pick}.\n"
        f"The host opens Door {revealed} — empty!\n"
        f"Remaining doors: {remaining}\n\n"
        f"Reminder:\n"
        f"{stay_line}\n"
        f"{switch_line}\n"
        f"{lose_line}\n\n"
        f"Round 1/{total_rounds}. Switch or stay?"
    )


# --- Metrics ---

@metric
def mean_switches() -> Metric:
    """Average number of switches across samples."""

    def metric_fn(scores: list[SampleScore]) -> float:
        switches = [
            s.score.metadata["switches"]
            for s in scores
            if s.score.metadata and "switches" in s.score.metadata
        ]
        return sum(switches) / len(switches) if switches else 0.0

    return metric_fn


@metric
def mean_reward() -> Metric:
    """Average reward across samples."""

    def metric_fn(scores: list[SampleScore]) -> float:
        rewards = [
            s.score.metadata["reward"]
            for s in scores
            if s.score.metadata and "reward" in s.score.metadata
        ]
        return sum(rewards) / len(rewards) if rewards else 0.0

    return metric_fn


@metric
def switch_rate() -> Metric:
    """Fraction of games where model switched at least once."""

    def metric_fn(scores: list[SampleScore]) -> float:
        switched = [
            1 if s.score.metadata and s.score.metadata.get("switches", 0) > 0 else 0
            for s in scores
        ]
        return sum(switched) / len(switched) if switched else 0.0

    return metric_fn


@metric
def mean_cashout_round() -> Metric:
    """Average round at which model cashed out or lost."""

    def metric_fn(scores: list[SampleScore]) -> float:
        rounds = [
            s.score.metadata["cashout_round"]
            for s in scores
            if s.score.metadata and "cashout_round" in s.score.metadata
        ]
        return sum(rounds) / len(rounds) if rounds else 0.0

    return metric_fn


@metric
def win_rate() -> Metric:
    """Fraction of games where model won (cash out or correct switch)."""

    def metric_fn(scores: list[SampleScore]) -> float:
        wins = [
            1 if s.score.metadata and s.score.metadata.get("won", False) else 0
            for s in scores
        ]
        return sum(wins) / len(wins) if wins else 0.0

    return metric_fn


# --- Solver ---

@solver
def setup_game_state() -> Solver:
    """Inject game metadata into store."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        state.store.set("winning_door", meta["winning_door"])
        state.store.set("n_doors", meta["n_doors"])
        state.store.set("total_rounds", meta["total_rounds"])
        state.store.set("framing", meta["framing"])
        state.store.set("payouts", meta["payouts"])
        state.store.set("mode", meta.get("mode", "standard"))
        state.store.set("game_started", False)
        return state

    return solve


# --- Dataset ---

def generate_samples(
    n: int = 50,
    n_doors: int = 3,
    framing: str = "money",
    final_reward: float = 1000,
    scale: float = 2.0,
    mode: str = "standard",
) -> list:
    """Generate game instances."""
    from inspect_ai.dataset import Sample

    total_rounds = n_doors - 2
    framing_data = FRAMINGS[framing]
    payouts = build_payouts(total_rounds, final_reward, scale)
    samples = []
    for i in range(n):
        winning_door = random.randint(1, n_doors)
        samples.append(
            Sample(
                input=(
                    f"Play the Monty Hall game with {n_doors} doors. "
                    f"Call make_decision('start') to begin, then 'switch' or 'stay' each round. "
                    f"{total_rounds} round(s) of decisions."
                ),
                target="switch",
                metadata={
                    "winning_door": winning_door,
                    "n_doors": n_doors,
                    "total_rounds": total_rounds,
                    "framing": framing_data,
                    "payouts": payouts,
                    "mode": mode,
                },
                id=str(i),
            )
        )
    return samples
