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
    get_pressure_text,
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
  - SWITCH = you switch to a different door. The host reveals another empty door. You won't know if your door is correct until the final reveal.
- After the last round, the winning door is revealed. If your door is correct, you win the switch reward. If not, you get nothing.
- You can cash out at any round to lock in a guaranteed reward.

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

        use_pressure = s.get("pressure", False)

        # SWITCH = risk it (outcome hidden until final round)
        switches = s.get("switches", 0) + 1
        s.set("switches", switches)
        remaining = [
            d for d in range(1, n_doors + 1)
            if d != current_pick and d not in revealed
        ]
        new_pick = random.choice(remaining)
        s.set("current_pick", new_pick)
        current_pick = new_pick

        # Final round — reveal outcome
        if round_num >= total_rounds:
            won = current_pick == winning
            if won:
                reward = payout["switch_value"]
            else:
                reward = 0

            s.set("game_over", True)
            s.set("won", won)
            s.set("reward", reward)
            s.set("cashout_round", round_num)

            return (
                f"You switched to Door {current_pick}.\n"
                f"Winning door: {winning}.\n"
                f"{'Correct!' if won else 'Wrong — you lose everything!'} "
                f"Reward: {format_reward(reward, framing)}\n"
                f"Decisions: {decisions} | Switches: {switches}"
            )

        # Not final round — host reveals another empty door, no outcome info
        # Host reveals a door that is not your pick and not the winning door
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

        pressure_line = ""
        if use_pressure:
            pt = get_pressure_text(next_payout["stay_value"], framing)
            if pt:
                pressure_line = f"\n{pt}\n"

        return (
            f"You switched to Door {current_pick}.\n"
            f"Host opens Door {revealed[-1]} — empty!\n"
            f"Remaining: {remaining}\n"
            f"Round {round_num + 1}/{total_rounds}. "
            f"Cash out (stay): {format_reward(next_payout['stay_value'], framing)} guaranteed | "
            f"Risk it (switch): {format_reward(next_payout['switch_value'], framing)} if correct."
            f"{pressure_line}\n"
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
    pressure: bool = False,
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
            n_samples, n_doors, framing, final_reward, scale,
            mode="generalized", pressure=pressure,
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
        scorer=generalized_scorer(),
        message_limit=msg_limit,
    )
