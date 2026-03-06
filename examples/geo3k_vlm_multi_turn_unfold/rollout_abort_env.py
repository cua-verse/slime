"""Multi-turn unfold rollout with env abort support.

Based on rollout.py, with the following additions:
  1. Cooperative abort check: generate() checks state.aborted each turn.
  2. Env registry: envs are registered/unregistered so abort() can clean them up.
  3. Custom abort(): cleans up all registered envs before SGLang abort.
  4. Custom generate_rollout: uses the local abort() instead of sglang_rollout.abort().

Usage:
  --custom-generate-function-path examples.geo3k_vlm_multi_turn_unfold.rollout_abort_env.generate
  --rollout-function-path examples.geo3k_vlm_multi_turn_unfold.rollout_abort_env.generate_rollout
  --custom-convert-samples-to-train-data-path examples.geo3k_vlm_multi_turn_unfold.rollout.convert_samples_to_train_data
"""

from __future__ import annotations

import asyncio
import logging
import re
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from packaging.version import parse
from tqdm import tqdm

import sglang_router

from examples.geo3k_vlm_multi_turn.rollout import (
    _build_env,
    _encode_observation_for_generation,
    _load_env_module,
    _merge_multimodal_train_inputs,
)
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import (
    GenerateState,
    eval_rollout,
)
from slime.utils.async_utils import run
from slime.utils.http_utils import get, post
from slime.utils.misc import load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env registry (module-level)
# ---------------------------------------------------------------------------

_active_envs: set = set()


def register_env(env):
    _active_envs.add(env)


def unregister_env(env):
    _active_envs.discard(env)


# ---------------------------------------------------------------------------
# Think-stripping utilities (copied from rollout.py)
# ---------------------------------------------------------------------------

_THINK_COMPLETE_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
_THINK_ORPHAN_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove all <think>...</think> blocks (complete and truncated)."""
    text = _THINK_COMPLETE_RE.sub("", text)
    text = _THINK_ORPHAN_CLOSE_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Internal helpers (copied from rollout.py)
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
    """Build a training Sample for one turn."""
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
        reward=None,
        status=status,
        multimodal_train_inputs=mm_train,
        metadata={**(original_sample.metadata or {}), "turn_idx": turn_idx},
    )


# ---------------------------------------------------------------------------
# Generate with cooperative abort check + env registry
# ---------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params: dict) -> list[Sample]:
    """Unfold multi-turn rollout with env abort support.

    Differences from rollout.py generate():
      - Checks state.aborted at each turn start and after env.step().
      - Registers/unregisters env in the module-level registry.
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
    register_env(env)

    # ------------------------------------------------------------------
    # Initial prompt encoding
    # ------------------------------------------------------------------
    prompt_text: str = sample.prompt

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
    stripped_context_ids: list[int] = list(prompt_ids)
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
            # ---- Cooperative abort check --------------------------------
            if state.aborted:
                break

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
                list(stripped_context_ids),
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

            # ---- Cooperative abort check after env.step() ----------------
            if state.aborted:
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
        if state.aborted and turn_samples:
            turn_samples[-1].status = Sample.Status.ABORTED
        elif turn_samples and turn_samples[-1].status == Sample.Status.PENDING:
            turn_samples[-1].status = Sample.Status.COMPLETED

        return turn_samples

    finally:
        unregister_env(env)
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Abort with env cleanup (based on sglang_rollout.abort)
# ---------------------------------------------------------------------------


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    """Extended abort: clean up registered envs, then do standard SGLang abort."""
    aborted_samples = []

    state = GenerateState(args)
    if state.aborted:
        # Already aborted, just wait for pendings.
        while state.pendings:
            done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        return aborted_samples

    state.aborted = True

    # ---- Actively clean up all registered envs ----
    for env in list(_active_envs):
        try:
            if hasattr(env, "abort") and callable(env.abort):
                env.abort()
            else:
                env.close()
        except Exception:
            logger.warning("Failed to abort env %s", env, exc_info=True)

    # ---- Standard SGLang abort (from sglang_rollout.abort) ----
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # Wait for all pending tasks to finish
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


# ---------------------------------------------------------------------------
# Custom generate_rollout_async (based on sglang_rollout.generate_rollout_async)
# Only difference: calls local abort() instead of sglang_rollout.abort()
# ---------------------------------------------------------------------------


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    assert args.rollout_global_dataset

    state = GenerateState(args)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # Use local abort with env cleanup
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
