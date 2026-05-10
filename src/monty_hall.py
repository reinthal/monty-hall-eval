"""Monty Hall eval — generalized to N doors, multi-turn, stateful.

Evaluates:
1. Switch rate — does the model discover optimal strategy?
2. Expected reward — does the model maximize EV given asymmetric payoffs?

The reward framing is pluggable (money, lives, etc.) to test whether
framing affects decision-making under identical probabilities.
"""

import random

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
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
from inspect_ai.solver import Generate, TaskState, generate, solver, Solver, system_message, use_tools
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store


# --- Reward framings ---

FRAMINGS = {
    "money": {
        "stay_reward": "$500 USD",
        "switch_reward": "$1000 USD",
        "stay_value": 500,
        "switch_value": 1000,
        "lose_value": 0,
        "context": "You are a contestant on a game show.",
    },
    "lives": {
        "stay_reward": "saving 500 lives",
        "switch_reward": "saving 1000 lives",
        "stay_value": 500,
        "switch_value": 1000,
        "lose_value": 0,
        "context": "You are a medical AI allocating emergency resources.",
    },
}


SYSTEM_PROMPT = """You are playing a generalized Monty Hall game.

{context}

Rules:
- There are {n_doors} doors. Behind one is the prize, the rest are empty.
- You pick a door. The host reveals an empty door.
- Each round: decide 'switch' or 'stay'.
- The host reveals empty doors until 2 remain, then you make a final choice.

Payoffs:
- If you STAY with your current door and it's correct: {stay_reward}
- If you SWITCH to the winning door: {switch_reward}
- Wrong door: nothing.

Use the tools to play. Call make_decision('start') to begin."""


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

        return (
            f"Round {round_num}/{total_rounds}\n"
            f"Total doors: {n_doors}\n"
            f"Your current pick: Door {picked}\n"
            f"Revealed empty doors: {revealed}\n"
            f"Remaining doors: {remaining}\n"
            f"Payoffs — stay & correct: {framing['stay_reward']} | "
            f"switch & correct: {framing['switch_reward']} | wrong: nothing"
        )

    return run


@tool
def make_decision() -> Tool:
    """Play the game: 'start' to begin, 'switch' or 'stay' each round."""

    async def run(decision: str) -> str:
        """Make your decision.

        Args:
            decision: One of 'start', 'switch', or 'stay'.
        """
        s = store()
        decision = decision.lower().strip()

        if s.get("game_over", False):
            return "Game is over. No more decisions needed."

        if decision == "start":
            if s.get("game_started", False):
                return "Game already started. Use 'switch' or 'stay'."
            s.set("game_started", True)
            n_doors = s.get("n_doors")
            initial_pick = random.randint(1, n_doors)
            s.set("current_pick", initial_pick)
            s.set("initial_pick", initial_pick)
            s.set("revealed_doors", [])
            s.set("round", 1)
            s.set("decisions", [])
            s.set("switches", 0)

            # Reveal first empty door
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
            return (
                f"Game started! {n_doors} doors.\n"
                f"You picked Door {initial_pick}.\n"
                f"The host opens Door {revealed} — empty!\n"
                f"Remaining doors: {remaining}\n\n"
                f"Reminder:\n"
                f"- Stay with Door {initial_pick} and win: {framing['stay_reward']}\n"
                f"- Switch to correct door: {framing['switch_reward']}\n"
                f"- Wrong door: nothing.\n\n"
                f"Round 1/{total_rounds}. Switch or stay?"
            )

        if decision not in ("switch", "stay"):
            return "Invalid. Choose 'start', 'switch', or 'stay'."

        if not s.get("game_started", False):
            return "Game not started. Call with 'start' first."

        n_doors = s.get("n_doors")
        winning = s.get("winning_door")
        current_pick = s.get("current_pick")
        revealed = s.get("revealed_doors", [])
        round_num = s.get("round", 1)
        total_rounds = s.get("total_rounds")
        framing = s.get("framing")

        decisions = s.get("decisions", [])
        decisions.append(decision)
        s.set("decisions", decisions)

        if decision == "switch":
            switches = s.get("switches", 0) + 1
            s.set("switches", switches)
            remaining = [
                d for d in range(1, n_doors + 1)
                if d != current_pick and d not in revealed
            ]
            new_pick = random.choice(remaining)
            s.set("current_pick", new_pick)
            current_pick = new_pick

        # Game over?
        if round_num >= total_rounds:
            won = current_pick == winning
            switches = s.get("switches", 0)
            ever_switched = switches > 0
            if won and not ever_switched:
                reward = framing["stay_value"]
            elif won and ever_switched:
                reward = framing["switch_value"]
            else:
                reward = framing["lose_value"]

            s.set("game_over", True)
            s.set("won", won)
            s.set("reward", reward)

            return (
                f"Final decision: {decision}. Your door: {current_pick}.\n"
                f"Winning door: {winning}.\n"
                f"{'Correct!' if won else 'Wrong.'} Reward: {reward}\n"
                f"Decisions: {decisions} | Switches: {switches}"
            )

        # Reveal another empty door
        revealable = [
            d for d in range(1, n_doors + 1)
            if d != current_pick and d != winning and d not in revealed
        ]
        if revealable:
            new_revealed = random.choice(revealable)
            revealed.append(new_revealed)
            s.set("revealed_doors", revealed)

        s.set("round", round_num + 1)
        remaining = [d for d in range(1, n_doors + 1) if d not in revealed]

        return (
            f"You chose to {decision}. Current door: {current_pick}.\n"
            f"Host opens Door {revealed[-1]} — empty!\n"
            f"Remaining: {remaining}\n"
            f"Round {round_num + 1}/{total_rounds}. Switch or stay?"
        )

    return run


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


@scorer(metrics=[switch_rate(), mean_switches(), mean_reward()])
def monty_hall_scorer():
    """Score: correct if switched, track switches and reward."""

    async def score(state: TaskState, target: Target) -> Score:
        switches = state.store.get("switches", 0)
        reward = state.store.get("reward", 0)
        won = state.store.get("won", False)
        decisions = state.store.get("decisions", [])

        return Score(
            value=CORRECT if switches > 0 else INCORRECT,
            explanation=f"decisions={decisions} switches={switches} reward={reward} won={won}",
            metadata={"switches": switches, "reward": reward, "won": won},
        )

    return score


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
        state.store.set("game_started", False)
        return state

    return solve


# --- Dataset ---

def generate_samples(n: int = 50, n_doors: int = 3, framing: str = "money") -> list[Sample]:
    """Generate game instances."""
    total_rounds = n_doors - 2
    framing_data = FRAMINGS[framing]
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
                },
                id=str(i),
            )
        )
    return samples


# --- Tasks ---

@task
def monty_hall(n_doors: int = 3, n_samples: int = 50, framing: str = "money") -> Task:
    """Monty Hall multi-turn agentic eval."""
    framing_data = FRAMINGS[framing]
    total_rounds = n_doors - 2
    # Budget: start + rounds of switch/stay + check_status calls + overhead
    msg_limit = 10 + total_rounds * 4
    return Task(
        dataset=MemoryDataset(samples=generate_samples(n_samples, n_doors, framing)),
        solver=[
            system_message(SYSTEM_PROMPT.format(
                context=framing_data["context"],
                n_doors=n_doors,
                stay_reward=framing_data["stay_reward"],
                switch_reward=framing_data["switch_reward"],
            )),
            setup_game_state(),
            use_tools(check_status(), make_decision(), tool_choice="any"),
            generate(tool_calls="loop"),
        ],
        scorer=monty_hall_scorer(),
        message_limit=msg_limit,
    )
