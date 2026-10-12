from __future__ import annotations

from typing import Any, TypedDict


class ModelInput(TypedDict):
    """One policy request containing numeric observation and raw language."""

    observation: dict[str, Any]
    instruction: str


def build_model_input(
    observation: dict[str, Any],
    instruction: str,
) -> ModelInput:
    """Package simulator output for a language-conditioned policy.

    The instruction intentionally stays outside ManiSkill's numeric Gym
    observation space. It is joined with the observation only at the policy
    boundary, so adapters can tokenize the original text themselves.
    """

    if not isinstance(observation, dict):
        raise TypeError(
            "observation must be a dictionary, got "
            f"{type(observation).__name__}."
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")
    return {
        "observation": observation,
        "instruction": instruction,
    }


def instruction_from_env(env: Any) -> str:
    """Read the current episode instruction through the public wrapper API."""

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            instruction = get_wrapper_attr("current_instruction")
        except AttributeError:
            instruction = None
    else:
        instruction = getattr(env, "current_instruction", None)

    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError(
            "The environment must expose a non-empty current_instruction "
            "after reset()."
        )
    return instruction
