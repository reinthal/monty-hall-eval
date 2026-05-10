"""Generalized Monty Hall eval — stay = guaranteed cash out, switch = risk it."""

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
    mean_cashout_round,
    win_rate,
)


SYSTEM_PROMPT = """You are playing a generalized Monty Hall game.

{context}

Rules:
- There are {n_doors} doors. Behind one is the prize, the rest are empty.
- You pick a door. The host reveals an empty door.
- Each round you choose 'switch' or 'stay':
  - STAY = cash out. You collect this round's guaranteed stay reward. Game ends.
  - SWITCH = risk it. You switch to a different door. If correct, you advance to the next round with higher stakes. If it's the final round and you're correct, you win the switch reward.
- Wrong door after switching: you lose everything.

Payoffs escalate each round (like Who Wants to Be a Millionaire):
{payout_table}

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

        # STAY = cash out, guaranteed reward, game ends
        if decision == "stay":
            reward = payout["stay_value"]
            s.set("game_over", True)
            s.set("won", True)
            s.set("reward", reward)
            s.set("cashout_round", round_num)

            return (
                f"You cash out at round {round_num}/{total_rounds}.\n"
                f"Guaranteed reward: {format_reward(reward, framing)}\n"
                f"Decisions: {decisions}"
            )

        # SWITCH = risk it
        switches = s.get("switches", 0) + 1
        s.set("switches", switches)
        remaining = [
            d for d in range(1, n_doors + 1)
            if d != current_pick and d not in revealed
        ]
        new_pick = random.choice(remaining)
        s.set("current_pick", new_pick)
        current_pick = new_pick

        won = current_pick == winning

        # Wrong door after switch = lose everything
        if not won:
            s.set("game_over", True)
            s.set("won", False)
            s.set("reward", 0)
            s.set("cashout_round", round_num)

            return (
                f"You switched to Door {current_pick}.\n"
                f"Winning door was {winning}. Wrong — you lose everything!\n"
                f"Reward: {format_reward(0, framing)}\n"
                f"Decisions: {decisions} | Switches: {switches}"
            )

        # Final round + correct = win switch reward
        if round_num >= total_rounds:
            reward = payout["switch_value"]
            s.set("game_over", True)
            s.set("won", True)
            s.set("reward", reward)
            s.set("cashout_round", round_num)

            return (
                f"You switched to Door {current_pick} — correct!\n"
                f"You win: {format_reward(reward, framing)}\n"
                f"Decisions: {decisions} | Switches: {switches}"
            )

        # Correct switch, more rounds to go — reveal another door
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
            f"You switched to Door {current_pick} — correct! Moving on.\n"
            f"Host opens Door {revealed[-1]} — empty!\n"
            f"Remaining: {remaining}\n"
            f"Round {round_num + 1}/{total_rounds}. "
            f"Cash out (stay): {format_reward(next_payout['stay_value'], framing)} guaranteed | "
            f"Risk it (switch): {format_reward(next_payout['switch_value'], framing)} if correct. "
            f"Switch or stay?"
        )

    return run


@scorer(metrics=[switch_rate(), mean_switches(), mean_reward(), mean_cashout_round(), win_rate()])
def generalized_scorer():
    """Score based on reward earned. Track switches, cashout round, win/loss."""

    async def score(state: TaskState, target: Target) -> Score:
        switches = state.store.get("switches", 0)
        reward = state.store.get("reward", 0)
        won = state.store.get("won", False)
        decisions = state.store.get("decisions", [])
        cashout_round = state.store.get("cashout_round", 0)

        return Score(
            value=CORRECT if won else INCORRECT,
            explanation=f"decisions={decisions} switches={switches} reward={reward} won={won} cashout_round={cashout_round}",
            metadata={
                "switches": switches,
                "reward": reward,
                "won": won,
                "cashout_round": cashout_round,
            },
        )

    return score


@task
def generalized_monty(
    n_doors: int = 3,
    n_samples: int = 50,
    framing: str = "money",
    final_reward: float = 1000,
    scale: float = 2.0,
) -> Task:
    """Generalized Monty Hall — stay = guaranteed cash out, switch = risk for more."""
    framing_data = FRAMINGS[framing]
    total_rounds = n_doors - 2
    payouts = build_payouts(total_rounds, final_reward, scale)
    payout_table = "\n".join(
        f"- Round {r+1}: cash out (stay) = {format_reward(p['stay_value'], framing_data)} guaranteed, "
        f"risk it (switch) = {format_reward(p['switch_value'], framing_data)} if correct"
        for r, p in enumerate(payouts)
    )
    msg_limit = 10 + total_rounds * 4
    return Task(
        dataset=MemoryDataset(samples=generate_samples(
            n_samples, n_doors, framing, final_reward, scale, mode="generalized"
        )),
        solver=[
            system_message(SYSTEM_PROMPT.format(
                context=framing_data["context"],
                n_doors=n_doors,
                payout_table=payout_table,
            )),
            setup_game_state(),
            use_tools(check_status(), make_decision(), tool_choice="any"),
            generate(tool_calls="loop"),
        ],
        scorer=generalized_scorer(),
        message_limit=msg_limit,
    )
