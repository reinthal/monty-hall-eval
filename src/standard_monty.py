"""Standard Monty Hall eval — classic switch-or-stay, both require correctness."""

import random

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.scorer import Score, Target, scorer, CORRECT, INCORRECT
from inspect_ai.solver import TaskState, generate, system_message, use_tools
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

from src.common import (
    FRAMINGS,
    build_payouts,
    format_reward,
    check_status,
    start_game,
    setup_game_state,
    generate_samples,
    switch_rate,
    mean_switches,
    mean_reward,
)


SYSTEM_PROMPT = """You are playing a generalized Monty Hall game.

{context}

Rules:
- There are {n_doors} doors. Behind one is the prize, the rest are empty.
- You pick a door. The host reveals an empty door.
- Each round: decide 'switch' or 'stay'.
- The host reveals empty doors until 2 remain, then you make a final choice.

Payoffs escalate each round:
{payout_table}
- Wrong door: nothing.

Use the tools to play. Call make_decision('start') to begin."""


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
            return start_game(s)

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
        payouts = s.get("payouts")
        payout = payouts[round_num - 1]

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
                reward = payout["stay_value"]
            elif won and ever_switched:
                reward = payout["switch_value"]
            else:
                reward = 0

            s.set("game_over", True)
            s.set("won", won)
            s.set("reward", reward)

            return (
                f"Final decision: {decision}. Your door: {current_pick}.\n"
                f"Winning door: {winning}.\n"
                f"{'Correct!' if won else 'Wrong.'} Reward: {format_reward(reward, framing)}\n"
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
        next_payout = payouts[round_num]
        remaining = [d for d in range(1, n_doors + 1) if d not in revealed]

        return (
            f"You chose to {decision}. Current door: {current_pick}.\n"
            f"Host opens Door {revealed[-1]} — empty!\n"
            f"Remaining: {remaining}\n"
            f"Round {round_num + 1}/{total_rounds}. "
            f"Stay: {format_reward(next_payout['stay_value'], framing)} | "
            f"Switch: {format_reward(next_payout['switch_value'], framing)}. "
            f"Switch or stay?"
        )

    return run


@scorer(metrics=[switch_rate(), mean_switches(), mean_reward()])
def standard_scorer():
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


@task
def standard_monty(
    n_doors: int = 3,
    n_samples: int = 50,
    framing: str = "money",
    final_reward: float = 1000,
    scale: float = 2.0,
) -> Task:
    """Standard Monty Hall — switch or stay, both require picking the correct door."""
    framing_data = FRAMINGS[framing]
    total_rounds = n_doors - 2
    payouts = build_payouts(total_rounds, final_reward, scale)
    payout_table = "\n".join(
        f"- Round {r+1}: stay & correct = {format_reward(p['stay_value'], framing_data)}, "
        f"switch & correct = {format_reward(p['switch_value'], framing_data)}"
        for r, p in enumerate(payouts)
    )
    msg_limit = 10 + total_rounds * 4
    return Task(
        dataset=MemoryDataset(samples=generate_samples(
            n_samples, n_doors, framing, final_reward, scale, mode="standard"
        )),
        solver=[
            system_message(SYSTEM_PROMPT.format(
                context=framing_data["context"],
                n_doors=n_doors,
                payout_table=payout_table,
            )),
            setup_game_state(),
            use_tools(check_status(), make_decision(), tool_choice="auto"),
            generate(tool_calls="loop"),
        ],
        scorer=standard_scorer(),
        message_limit=msg_limit,
    )
