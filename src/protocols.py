"""Solver -> critic -> revision protocol plus baselines."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .axis import SteeringHook
from .generate import GenConfig, batched_generate

SOLVER_PROMPT = """Solve the following math word problem. Think step by step, then give the final answer on a separate line as:

Final answer: <answer>

Problem:
{question}"""

CRITIC_PROMPT = """You are checking another model's solution.

Problem:
{question}

Proposed solution:
{initial_answer}

Check the solution carefully. Identify any mistake if present. If the solution is correct, say it is correct. Be concise."""

REVISION_PROMPT = """You previously solved this problem and received a critique.

Problem:
{question}

Your initial solution:
{initial_answer}

Critique:
{critique}

Revise your solution if needed. Think step by step, then give the final answer on a separate line as:

Final answer: <answer>"""

SELF_REVISION_PROMPT = """You previously solved this problem.

Problem:
{question}

Your initial solution:
{initial_answer}

Look at the solution again. Revise if needed. Think step by step, then give the final answer on a separate line as:

Final answer: <answer>"""


@dataclass
class Transcript:
    example_id: str
    question: str
    gold_number: float
    alpha_solver: float
    alpha_critic: float
    initial_text: str
    critique_text: str
    final_text: str


def _solver_messages(question: str):
    return [{"role": "user", "content": SOLVER_PROMPT.format(question=question)}]


def _critic_messages(question: str, initial_answer: str):
    return [
        {
            "role": "user",
            "content": CRITIC_PROMPT.format(question=question, initial_answer=initial_answer),
        }
    ]


def _revision_messages(question: str, initial_answer: str, critique: str):
    return [
        {
            "role": "user",
            "content": REVISION_PROMPT.format(
                question=question, initial_answer=initial_answer, critique=critique
            ),
        }
    ]


def _self_revision_messages(question: str, initial_answer: str):
    return [
        {
            "role": "user",
            "content": SELF_REVISION_PROMPT.format(
                question=question, initial_answer=initial_answer
            ),
        }
    ]


def _maybe_steer(model, axis: torch.Tensor | None, layer_idx: int, alpha: float):
    if axis is None or alpha == 0.0:
        return contextlib.nullcontext()
    vec = axis[layer_idx] if axis.dim() == 2 else axis
    return SteeringHook(model, vec, layer_idx, alpha)


def run_two_agent(
    examples,
    model,
    tokenizer,
    axis: torch.Tensor,
    layer_idx: int,
    alpha_solver: float,
    alpha_critic: float,
    solver_gen: GenConfig,
    critic_gen: GenConfig,
    batch_size: int = 16,
) -> list[Transcript]:
    """Solver pass (α_s) -> Critic pass (α_c) -> Revision pass (α_s)."""
    questions = [ex.question for ex in examples]

    solver_msgs = [_solver_messages(q) for q in questions]
    initials = batched_generate(
        model,
        tokenizer,
        solver_msgs,
        solver_gen,
        steering_ctx=_maybe_steer(model, axis, layer_idx, alpha_solver),
        batch_size=batch_size,
    )

    critic_msgs = [_critic_messages(q, ia) for q, ia in zip(questions, initials)]
    critiques = batched_generate(
        model,
        tokenizer,
        critic_msgs,
        critic_gen,
        steering_ctx=_maybe_steer(model, axis, layer_idx, alpha_critic),
        batch_size=batch_size,
    )

    revision_msgs = [
        _revision_messages(q, ia, cr) for q, ia, cr in zip(questions, initials, critiques)
    ]
    finals = batched_generate(
        model,
        tokenizer,
        revision_msgs,
        solver_gen,
        steering_ctx=_maybe_steer(model, axis, layer_idx, alpha_solver),
        batch_size=batch_size,
    )

    return [
        Transcript(
            example_id=ex.id,
            question=ex.question,
            gold_number=ex.gold_number,
            alpha_solver=alpha_solver,
            alpha_critic=alpha_critic,
            initial_text=ia,
            critique_text=cr,
            final_text=fi,
        )
        for ex, ia, cr, fi in zip(examples, initials, critiques, finals)
    ]


def run_single_agent_direct(
    examples,
    model,
    tokenizer,
    solver_gen: GenConfig,
    batch_size: int = 16,
) -> list[Transcript]:
    questions = [ex.question for ex in examples]
    solver_msgs = [_solver_messages(q) for q in questions]
    finals = batched_generate(
        model, tokenizer, solver_msgs, solver_gen, steering_ctx=None, batch_size=batch_size
    )
    return [
        Transcript(
            example_id=ex.id,
            question=ex.question,
            gold_number=ex.gold_number,
            alpha_solver=0.0,
            alpha_critic=0.0,
            initial_text=fi,
            critique_text="",
            final_text=fi,
        )
        for ex, fi in zip(examples, finals)
    ]


def run_single_agent_self_revision(
    examples,
    model,
    tokenizer,
    solver_gen: GenConfig,
    batch_size: int = 16,
) -> list[Transcript]:
    questions = [ex.question for ex in examples]
    solver_msgs = [_solver_messages(q) for q in questions]
    initials = batched_generate(
        model, tokenizer, solver_msgs, solver_gen, steering_ctx=None, batch_size=batch_size
    )
    rev_msgs = [_self_revision_messages(q, ia) for q, ia in zip(questions, initials)]
    finals = batched_generate(
        model, tokenizer, rev_msgs, solver_gen, steering_ctx=None, batch_size=batch_size
    )
    return [
        Transcript(
            example_id=ex.id,
            question=ex.question,
            gold_number=ex.gold_number,
            alpha_solver=0.0,
            alpha_critic=0.0,
            initial_text=ia,
            critique_text="",
            final_text=fi,
        )
        for ex, ia, fi in zip(examples, initials, finals)
    ]
