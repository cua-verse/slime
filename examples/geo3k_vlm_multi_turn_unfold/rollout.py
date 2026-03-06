"""Multi-turn rollout with think-stripping ("unfold" variant).

Key differences from geo3k_vlm_multi_turn:
  1. SGLang bug fix: uses "text" payload instead of "input_ids" to avoid the
     image-padding IndexError (pre-expanded image_pad tokens vs. 1 image).
  2. Think-stripping: each turn sees context where previous turns' <think>...</think>
     blocks have been removed.
  3. Returns list[Sample] (one per completed turn) for correct training alignment:
     the training-time context matches the generation-time context exactly.

Training sample structure for turn k:
  tokens           = context_k + resp_k
  where context_k  = prompt_ids + strip(resp_1) + obs_1 + ... + strip(resp_{k-1}) + obs_{k-1}
  loss_mask        = [1 × len(resp_k)]
  response_length  = len(resp_k)
  rollout_log_probs = actual_log_probs_k

  The entire context_k (including stripped history and observations) is treated
  as the "prompt" from the training framework's perspective, so that
  total_length = len(context_k) + response_length always holds.

convert_samples_to_train_data():
  Handles GRPO advantage normalization at the ROLLOUT level (not per-turn):
  all per-turn samples from the same rollout share the same advantage.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from examples.geo3k_vlm_multi_turn.rollout import (
    _build_env,
    _encode_observation_for_generation,
    _load_env_module,
    _merge_multimodal_train_inputs,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

# ---------------------------------------------------------------------------
# Think-stripping utilities
# ---------------------------------------------------------------------------

_THINK_COMPLETE_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
# Handles the case where <think> is in the context prefix and </think> is in the response.
# Qwen3 chat template appends <|im_start|>assistant\n<think>\n before generation,
# so the response starts inside a think block (no opening tag in response text).
_THINK_ORPHAN_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks (complete and truncated).

    Handles three cases:
    1. Complete blocks: <think>content</think> → removed.
    2. Orphan close: content</think> at start (think opened in context prefix) → removed.
    3. Truncated open: <think>content at end (generation cut off) → removed.
    """
    text = _THINK_COMPLETE_RE.sub("", text)
    text = _THINK_ORPHAN_CLOSE_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_text_inference(
    url: str,
    context_text: str,
    sampling_params: dict,
    image_data: list,
) -> tuple[str, list[int], list[float], str]:
    """Run SGLang inference with text payload (fixes input_ids + image_data bug)."""
    payload: dict = {
        "text": context_text,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text: str = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type: str = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


def _make_turn_sample(
    original_sample: Sample,
    context_ids: list[int],
    response_ids: list[int],
    response_log_probs: list[float],
    response_text: str,
    status: Sample.Status,
    turn_idx: int,
    mm_train_buffer: list[dict | None],
) -> Sample:
    """Build a training Sample for one turn.

    context_ids is treated as the full "prompt" by the training framework:
      total_length = len(context_ids) + response_length
    Only resp_k tokens are trained on (loss_mask = all-ones of len response_ids).
    """
    tokens = context_ids + response_ids
    response_length = len(response_ids)
    loss_mask = [1] * response_length
    rollout_log_probs = list(response_log_probs)

    mm_train = _merge_multimodal_train_inputs(mm_train_buffer)

    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        prompt=original_sample.prompt,
        tokens=tokens,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        response_length=response_length,
        response=response_text,
        label=original_sample.label,
        reward=None,  # assigned later by reward model / convert_samples
        status=status,
        multimodal_train_inputs=mm_train,
        metadata={**(original_sample.metadata or {}), "turn_idx": turn_idx},
    )


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params: dict) -> list[Sample]:
    """Unfold multi-turn rollout returning one Sample per completed turn.

    SGLang bug fix: uses "text" payload so image placeholders are not pre-expanded.
    Think-stripping: each turn's context hides previous turns' <think> blocks.
    """
    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for unfold rollouts."
    )

    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    max_turns: int = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set in custom config.")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    processor = state.processor
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)

    # ------------------------------------------------------------------
    # Initial prompt encoding
    # ------------------------------------------------------------------
    prompt_text: str = sample.prompt  # text with <|image_pad|> placeholder (1 per image)

    if processor:
        proc_out = processor(text=prompt_text, **(sample.multimodal_inputs or {}))
        prompt_ids: list[int] = list(proc_out["input_ids"][0])
        init_mm_train = {
            k: v for k, v in proc_out.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        init_mm_train = None

    initial_image_data: list = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        initial_image_data = [
            encode_image_for_rollout_engine(img)
            for img in sample.multimodal_inputs["images"]
        ]

    prompt_len = len(prompt_ids)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    # Token-level context (used to build per-turn training tokens):
    #   prompt_ids + strip(resp_1) + obs_1_ids + ... + strip(resp_{k-1}) + obs_{k-1}_ids
    stripped_context_ids: list[int] = list(prompt_ids)

    # Text-level context (sent as "text" to SGLang):
    #   prompt_text + strip(resp_1_text) + obs_1_text + ...
    stripped_context_text: str = prompt_text

    current_image_data: list = list(initial_image_data)
    mm_train_buffer: list[dict | None] = [init_mm_train] if init_mm_train else []

    # ------------------------------------------------------------------
    # Budget (max context length)
    # ------------------------------------------------------------------
    budget: int | None = None
    if getattr(args, "rollout_max_context_len", None) is not None:
        budget = args.rollout_max_context_len - prompt_len

    cur_sampling_params = sampling_params.copy()

    turn_samples: list[Sample] = []

    try:
        env.reset()

        if budget is not None and budget <= 0:
            s = _make_turn_sample(
                sample, stripped_context_ids, [], [], "",
                Sample.Status.TRUNCATED, 0, mm_train_buffer,
            )
            return [s]

        for turn_idx in range(max_turns):
            if budget is not None:
                cur_sampling_params = {**cur_sampling_params, "max_new_tokens": budget}

            # ---- Generate -----------------------------------------------
            response_text, response_ids, response_log_probs, finish_type = (
                await _run_text_inference(
                    url, stripped_context_text, cur_sampling_params, current_image_data
                )
            )

            # ---- Determine status ----------------------------------------
            if finish_type == "length":
                turn_status = Sample.Status.TRUNCATED
            elif finish_type == "abort":
                turn_status = Sample.Status.ABORTED
            else:
                turn_status = Sample.Status.COMPLETED

            # ---- Build per-turn training Sample --------------------------
            turn_sample = _make_turn_sample(
                sample,
                list(stripped_context_ids),  # snapshot BEFORE appending this turn
                response_ids,
                response_log_probs,
                response_text,
                turn_status,
                turn_idx,
                list(mm_train_buffer),
            )
            turn_samples.append(turn_sample)

            # Update budget
            if budget is not None:
                budget -= len(response_ids)

            # ---- Stop on finish type or budget ---------------------------
            if finish_type in ("length", "abort"):
                break
            if budget is not None and budget <= 0:
                break

            # ---- Environment step ----------------------------------------
            observation, done, _info = env.step(response_text)
            if done:
                turn_samples[-1].status = Sample.Status.COMPLETED
                break

            obs_message = env.format_observation(observation)
            obs_ids, obs_image_data, _obs_mm_inputs, obs_mm_train = (
                _encode_observation_for_generation(
                    tokenizer,
                    processor,
                    obs_message,
                    sample.metadata,
                    getattr(args, "apply_chat_template", True),
                    getattr(args, "apply_chat_template_kwargs", None),
                )
            )

            # Strip BOS from obs_ids if accidentally prepended
            bos_id = tokenizer.bos_token_id
            if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                obs_ids = obs_ids[1:]

            # Decode obs tokens to text for SGLang context
            obs_text: str = tokenizer.decode(obs_ids, skip_special_tokens=False)

            # Strip <think> from this response for the NEXT turn's context
            stripped_resp_text = strip_think(response_text)
            stripped_resp_ids = tokenizer.encode(stripped_resp_text, add_special_tokens=False)

            # ---- Update accumulators for next turn -----------------------
            stripped_context_ids = stripped_context_ids + stripped_resp_ids + list(obs_ids)
            stripped_context_text = stripped_context_text + stripped_resp_text + obs_text

            if obs_image_data:
                current_image_data = current_image_data + obs_image_data
            if obs_mm_train:
                mm_train_buffer = mm_train_buffer + [obs_mm_train]

            if budget is not None:
                budget -= len(obs_ids)
                if budget <= 0:
                    break

            if turn_idx + 1 >= max_turns:
                turn_samples[-1].status = Sample.Status.COMPLETED
                break

        # Final status fixup
        if turn_samples and turn_samples[-1].status == Sample.Status.PENDING:
            turn_samples[-1].status = Sample.Status.COMPLETED

        return turn_samples

    finally:
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Custom convert_samples_to_train_data
# ---------------------------------------------------------------------------


def convert_samples_to_train_data(args: Any, samples) -> dict:
    """Custom training-data converter for unfold multi-turn samples.

    Handles GRPO advantage at the ROLLOUT level:
      - Collects one terminal reward per rollout (last turn's reward)
      - Normalises within each prompt group (n_samples_per_prompt rollouts)
      - Assigns the same advantage to all per-turn Samples from that rollout
      - Flattens into a standard train_data dict

    Args:
        args: training args
        samples: list[list[Sample]] or list[Sample]
            The infrastructure passes list[list[Sample]] when generate() returns a list.
            We also handle list[Sample] (already-flat) as a fallback.
    """
    if not samples:
        return {}

    # ------------------------------------------------------------------
    # Normalise to list[list[Sample]]
    # ------------------------------------------------------------------
    if isinstance(samples[0], Sample):
        # Already flat — reconstruct rollout grouping from (group_index, index)
        rollout_map: dict[tuple, list[Sample]] = {}
        for s in samples:
            key = (s.group_index, s.index)
            rollout_map.setdefault(key, []).append(s)
        rollouts: list[list[Sample]] = list(rollout_map.values())
    else:
        rollouts = list(samples)  # list[list[Sample]]

    # ------------------------------------------------------------------
    # Collect terminal rewards (one per rollout)
    # ------------------------------------------------------------------
    terminal_rewards: list[float] = []
    for rollout in rollouts:
        r = rollout[-1].reward if rollout else 0.0
        terminal_rewards.append(float(r) if r is not None else 0.0)

    # ------------------------------------------------------------------
    # GRPO advantage normalisation at rollout level
    # ------------------------------------------------------------------
    n_samples = getattr(args, "n_samples_per_prompt", 1)
    if (
        getattr(args, "advantage_estimator", None) in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and getattr(args, "rewards_normalization", True)
    ):
        rewards_t = torch.tensor(terminal_rewards, dtype=torch.float)
        # Reshape to (n_groups, n_samples_per_prompt) if evenly divisible
        if len(rewards_t) % n_samples == 0:
            rewards_t = rewards_t.reshape(-1, n_samples)
        else:
            rewards_t = rewards_t.unsqueeze(0)

        mean = rewards_t.mean(dim=-1, keepdim=True)
        advantages_t = rewards_t - mean

        if (
            getattr(args, "advantage_estimator", None) in ("grpo", "gspo")
            and getattr(args, "grpo_std_normalization", True)
        ):
            std = advantages_t.std(dim=-1, keepdim=True)
            advantages_t = advantages_t / (std + 1e-6)

        advantages: list[float] = advantages_t.flatten().tolist()
    else:
        advantages = terminal_rewards

    # ------------------------------------------------------------------
    # Propagate advantage to all per-turn Samples of each rollout
    # ------------------------------------------------------------------
    for rollout, adv in zip(rollouts, advantages):
        for turn_sample in rollout:
            turn_sample.reward = adv

    # ------------------------------------------------------------------
    # Flatten
    # ------------------------------------------------------------------
    flat: list[Sample] = [s for rollout in rollouts for s in rollout]

    # ------------------------------------------------------------------
    # Assemble train_data dict
    # ------------------------------------------------------------------
    train_data: dict = {
        "tokens": [s.tokens for s in flat],
        "response_lengths": [s.response_length for s in flat],
        "rewards": [s.reward for s in flat],
        "raw_reward": [s.reward for s in flat],
        "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in flat],
        "sample_indices": [s.index for s in flat],
        "loss_masks": [s.loss_mask for s in flat],
    }

    if flat[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [s.rollout_log_probs for s in flat]

    if any(s.multimodal_train_inputs is not None for s in flat):
        train_data["multimodal_train_inputs"] = [s.multimodal_train_inputs for s in flat]

    return train_data
