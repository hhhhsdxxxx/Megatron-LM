#!/usr/bin/env python3
# Copyright (c) 2024. All rights reserved.
"""
Simulate Megatron-LM training data pipeline and dump every step to disk
so the data order can be compared with other training frameworks (e.g.
ant-pretrain / MaxText on TPU).

Output layout (aligned with ant-pretrain/scripts/dump_dataloader.py):

    <dump-output-dir>/
        metadata.json
        step_000000/
            host_0000.npz   # dp_rank 0: tokens, labels, loss_mask, …
            host_0001.npz   # dp_rank 1
            ...
        step_000001/
            host_0000.npz
            ...

Each host_XXXX.npz contains the **concatenation** of all micro-batches
for that step on that DP rank, with shape
``(num_microbatches * micro_batch_size, seq_length)``.

The script is **pure CPU, single process** – no GPU or distributed init
needed.  It loops over all DP ranks internally.

Usage (pass the same args you would pass to pretrain_gpt.py):

    python dump_dataloader.py \\
        <all original training args> \\
        --dump-output-dir  /path/to/dump \\
        --dump-num-steps   0          # 0 = dump all steps (default)
        --dump-dp-size     8          # total DP world size
        --dump-consumed-samples 0     # consumed samples to resume from (default 0)
"""

import argparse
import json
import os
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1.  Extra argument provider – dump-specific CLI flags
# ---------------------------------------------------------------------------

def _add_dump_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="dump-dataloader")
    group.add_argument(
        "--dump-output-dir",
        type=str,
        required=True,
        help="Root directory for dumped data.",
    )
    group.add_argument(
        "--dump-num-steps",
        type=int,
        default=0,
        help="Number of training steps to dump.  0 = all steps (default).",
    )
    group.add_argument(
        "--dump-dp-rank",
        type=int,
        default=-1,
        help="Data-parallel rank to simulate. "
             "-1 = dump ALL ranks (default).  0..dp_size-1 = single rank.",
    )
    group.add_argument(
        "--dump-dp-size",
        type=int,
        required=True,
        help="Total data-parallel world size.",
    )
    group.add_argument(
        "--dump-consumed-samples",
        type=int,
        default=0,
        help="Number of samples already consumed (for resumption, default 0).",
    )
    group.add_argument(
        "--dump-num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default 0, single-process).",
    )
    return parser


# ---------------------------------------------------------------------------
# 2.  Dummy tokenizer (no real tokenizer files needed)
# ---------------------------------------------------------------------------

class _DummyTokenizer:
    """Minimal tokenizer stub that satisfies GPTDatasetConfig / GPTDataset.

    The dataset pipeline only touches two attributes at runtime:
      - ``vocab_size``  → GPTDatasetConfig.__post_init__ (token_dtype_code)
      - ``eod``         → GPTDataset.__getitem__ (loss_mask / position_ids)

    Neither affects the *token content* or *data order* that we dump,
    so any reasonable value works.
    """

    def __init__(self, vocab_size: int = 200000, eod_id: int = 0):
        self.vocab_size = vocab_size
        self.eod = eod_id
        # MegatronDataset.__init__ probes these (all wrapped in try/except)
        self.pad = eod_id
        self.eos = eod_id
        # MegatronDataset serialises config objects via obj.unique_identifiers
        self.unique_identifiers = {"tokenizer": "DummyTokenizer", "vocab_size": vocab_size}

    @property
    def special_tokens_dict(self):
        return {}


# ---------------------------------------------------------------------------
# 3.  Lightweight initialisation (no GPU / no distributed)
# ---------------------------------------------------------------------------

def _init_megatron_for_dump():
    """Parse Megatron args, set global variables – all without touching
    CUDA, torch.distributed, or real tokenizer files.

    Returns the parsed ``args`` namespace (with dump-specific fields).
    """

    # ---- parse args (reuse Megatron's parser) ----
    from megatron.training.arguments import parse_args, validate_args

    args = parse_args(extra_args_provider=_add_dump_args, ignore_unknown_args=True)

    # ---- Force-disable tokenizer so nothing tries to load real files ----
    args.tokenizer_type = None
    args.tokenizer_model = None

    # Force single-process view so validate_args computes data_parallel_size
    # from the *dump* world size (which is 1 process pretending to be the
    # full cluster).
    args.rank = 0
    args.world_size = (
        args.dump_dp_size
        * args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )

    # Avoid validate_args checks that require CUDA --------------------------------
    # moe_grouped_gemm triggers torch.cuda.get_device_capability – disable it.
    if getattr(args, "moe_grouped_gemm", False):
        args.moe_grouped_gemm = False

    # Env var assertions for TP/CP with CUDA_DEVICE_MAX_CONNECTIONS – satisfy them.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    # validate_args (sets data_parallel_size, consumed_train_samples, etc.)
    validate_args(args, defaults={"tokenizer_type": "GPT2BPETokenizer"})

    # Override data_parallel_size to match dump args.
    args.data_parallel_size = args.dump_dp_size

    # ---- set global variables (skip tokenizer build — we use a dummy) ----
    from megatron.training.global_vars import set_global_variables
    set_global_variables(args, build_tokenizer=False)

    return args


# ---------------------------------------------------------------------------
# 4.  Build the dataset (reuse pretrain_gpt.py helpers)
# ---------------------------------------------------------------------------

def _build_train_dataset(args):
    """Return the *training* dataset only (valid/test are ignored)."""
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.training.utils import get_blend_and_blend_per_split

    # Use a dummy tokenizer — no real tokenizer files needed for dump
    tokenizer = _DummyTokenizer(
        vocab_size=getattr(args, "vocab_size", 200000),
        eod_id=0,
    )
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    # Load per-dataset sequence counts if configured
    sequences_per_dataset = None
    if getattr(args, "per_dataset_sequences_path", None) is not None:
        import json as _json
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = _json.load(f)

    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        # s3_cache_path has been renamed to object_storage_cache_path
        object_storage_cache_path=getattr(args, "object_storage_cache_path", None),
        # New fields that affect data ordering / alignment
        data_parallel_size=args.dump_dp_size,
        context_parallel_size=getattr(args, "context_parallel_size", 1),
        sequence_parallel_size=(
            getattr(args, "tensor_model_parallel_size", 1)
            * getattr(args, "sequence_parallel", 0)
        ),
        hybrid_context_parallel=getattr(args, "hybrid_context_parallel", False),
        # New optional fields
        multiple_validation_sets=getattr(args, "multiple_validation_sets", None),
        full_validation=getattr(args, "full_validation", None),
        mid_level_dataset_surplus=getattr(args, "mid_level_dataset_surplus", 0.005),
        allow_ambiguous_pad_tokens=getattr(args, "allow_ambiguous_pad_tokens", False),
        fast_cache_load=getattr(args, "dataloader_fast_cache_load", False),
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=getattr(args, "dataloader_defer_npy_index_mmap", False),
    )

    # Compute target number of training samples
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )

    print(f"> dataset target sizes: train={train_val_test_num_samples[0]}, "
          f"valid={train_val_test_num_samples[1]}, test={train_val_test_num_samples[2]}")

    # torch.distributed is *not* initialised → builder will take the
    # ``not torch.distributed.is_initialized()`` path, which simply calls
    # ``cls(*args)`` without barriers.
    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        lambda: True,  # is_built_on_rank – always True for single-process
        config,
    ).build()

    return train_ds


# ---------------------------------------------------------------------------
# 5.  Build the sampler & data-loader for a single DP rank
# ---------------------------------------------------------------------------

def _build_dataloader(dataset, args, dp_rank):
    """Construct the same DataLoader that Megatron would use at runtime
    for a given ``dp_rank``.

    Returns ``(dataloader, batch_size_per_step)`` where
    ``batch_size_per_step`` is the number of samples this rank produces
    per training step (= num_microbatches_per_step * micro_batch_size).
    """
    from megatron.training.datasets.data_samplers import MegatronPretrainingSampler

    consumed = args.dump_consumed_samples

    sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=args.dump_dp_size,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.dump_num_workers,
        pin_memory=False,
    )
    return loader


# ---------------------------------------------------------------------------
# 6.  Dump loop — one DP rank
# ---------------------------------------------------------------------------

def _dump_one_rank(args, dataset, dp_rank, total_steps, num_microbatches_per_step):
    """Dump all steps for a single DP rank.

    Output layout per step (aligned with ant-pretrain):
        <dump_output_dir>/step_NNNNNN/host_RRRR.npz

    Each .npz concatenates all micro-batches of the step and contains
    every field returned by the dataset (tokens, labels, loss_mask, …).
    """
    out_dir = args.dump_output_dir
    dataloader = _build_dataloader(dataset, args, dp_rank)

    step = 0
    mb_in_step = 0
    t_start = time.time()
    total_mb = 0

    # Accumulate micro-batches within a step
    step_accum: dict[str, list[np.ndarray]] = {}

    for batch in dataloader:
        # batch: dict[str, Tensor] from DataLoader (tokens, labels, loss_mask, …)
        for key, val in batch.items():
            arr = val.numpy()
            step_accum.setdefault(key, []).append(arr)

        mb_in_step += 1
        total_mb += 1

        if mb_in_step >= num_microbatches_per_step:
            # Flush accumulated micro-batches as one file
            step_dir = os.path.join(out_dir, f"step_{step:06d}")
            os.makedirs(step_dir, exist_ok=True)

            # Concatenate all micro-batches along batch dim
            merged = {k: np.concatenate(v, axis=0) for k, v in step_accum.items()}
            np.savez(os.path.join(step_dir, f"host_{dp_rank:04d}.npz"), **merged)

            step += 1
            mb_in_step = 0
            step_accum = {}

            if step % 100 == 0:
                elapsed = time.time() - t_start
                shapes = " ".join(f"{k}={v.shape}" for k, v in merged.items())
                print(
                    f"  [rank {dp_rank}] step {step}/{total_steps or '?'} "
                    f"({total_mb} micro-batches, {elapsed:.1f}s) {shapes}"
                )

            if total_steps is not None and step >= total_steps:
                break

    elapsed = time.time() - t_start
    print(
        f"> [rank {dp_rank}] dump complete: {step} steps, "
        f"{total_mb} micro-batches in {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("dump_dataloader.py – Megatron data pipeline offline dumper")
    print("=" * 60)

    args = _init_megatron_for_dump()

    # Determine which DP ranks to dump
    if args.dump_dp_rank >= 0:
        dp_ranks = [args.dump_dp_rank]
    else:
        dp_ranks = list(range(args.dump_dp_size))

    # Compute how many micro-batches form one training step per rank
    num_microbatches_per_step = args.global_batch_size // (
        args.micro_batch_size * args.dump_dp_size
    )
    assert num_microbatches_per_step >= 1, (
        f"global_batch_size={args.global_batch_size} is too small for "
        f"mbs={args.micro_batch_size} * dp={args.dump_dp_size}"
    )

    total_steps = args.dump_num_steps if args.dump_num_steps > 0 else None
    if total_steps is None and args.train_iters:
        total_steps = args.train_iters

    print(f"> config: dp_ranks={dp_ranks}, dp_size={args.dump_dp_size}, "
          f"mbs={args.micro_batch_size}, gbs={args.global_batch_size}, "
          f"seq_len={args.seq_length}, seed={args.seed}")
    print(f"> {num_microbatches_per_step} micro-batches/step, "
          f"total_steps={total_steps}")

    # Build dataset (shared across all DP ranks — same as real training)
    print("> building dataset …")
    train_ds = _build_train_dataset(args)
    assert train_ds is not None, "Failed to build training dataset"
    print(f"> training dataset size: {len(train_ds)} samples")

    # Write metadata
    out_dir = args.dump_output_dir
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "seq_length": args.seq_length,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "dp_ranks_dumped": dp_ranks,
        "dp_size": args.dump_dp_size,
        "num_microbatches_per_step": num_microbatches_per_step,
        "consumed_samples_start": args.dump_consumed_samples,
        "total_steps": total_steps,
        "train_iters": args.train_iters,
        "train_samples": args.train_samples,
        "data_path": args.data_path,
        "split": args.split,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Dump each DP rank — one process per rank for parallel I/O
    if len(dp_ranks) == 1:
        # Single rank: run in main process, no overhead
        print(f"> dumping dp_rank={dp_ranks[0]} …", flush=True)
        _dump_one_rank(args, train_ds, dp_ranks[0], total_steps, num_microbatches_per_step)
    else:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)  # fork to share mmap'd dataset

        procs = []
        for dp_rank in dp_ranks:
            print(f"> spawning process for dp_rank={dp_rank} …", flush=True)
            p = mp.Process(
                target=_dump_one_rank,
                args=(args, train_ds, dp_rank, total_steps, num_microbatches_per_step),
            )
            p.start()
            procs.append((dp_rank, p))

        # Wait for all ranks to finish
        for dp_rank, p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"ERROR: dp_rank={dp_rank} exited with code {p.exitcode}",
                      flush=True)

    print(f"> all done. output written to {out_dir}")


if __name__ == "__main__":
    main()
