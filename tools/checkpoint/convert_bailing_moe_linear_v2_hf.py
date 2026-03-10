#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Convert BailingMoE Linear V2 HF safetensors to Megatron legacy torch format.

Builds the Megatron model in-process (same pattern as loader_mixtral_hf.py),
copies weights from HF safetensors, and saves with torch.save.

Usage:
    python tools/checkpoint/convert_bailing_moe_linear_v2_hf.py \
        --input-dir /path/to/hf/safetensors \
        --output-dir /path/to/megatron/checkpoint
"""

import argparse
import glob as glob_module
import os
import sys
import types

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BailingMoE Linear V2 HF safetensors to Megatron legacy torch format.'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to HF safetensors checkpoint directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Megatron checkpoint')
    parser.add_argument('--iteration', type=int, default=23842,
                        help='Iteration number for the checkpoint')
    parser.add_argument('--target-expert-parallel-size', type=int, default=8,
                        help='Expert parallel size for checkpoint sharding (default: 8)')
    return parser.parse_args()


def load_hf_state_dict(input_dir):
    """Load all safetensors files and merge into one dict."""
    from safetensors.torch import load_file

    state_dict = {}
    sf_files = sorted(glob_module.glob(os.path.join(input_dir, '*.safetensors')))
    assert sf_files, f"No safetensors files found in {input_dir}"
    print(f"Loading {len(sf_files)} safetensors files...")
    for sf in sf_files:
        state_dict.update(load_file(sf, device='cpu'))
    print(f"Loaded {len(state_dict)} tensors from HF checkpoint")
    return state_dict


def initialize_megatron():
    """Set sys.argv with model args, parse, and initialize Megatron environment."""
    # Add repo root to path (following loader_mixtral_hf.py pattern)
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    ))

    # Override sys.argv with all model args matching CI script.
    # Key difference: --expert-model-parallel-size 1 (single-file checkpoint).
    sys.argv = [
        'convert_bailing_moe_linear_v2_hf.py',
        '--use-mcore-models',
        '--disable-bias-linear',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--mock-data',
        '--transformer-impl', 'transformer_engine',
        '--no-one-logger',
        # Model architecture (from ci_bailing_moe_linear_v2.sh)
        '--spec', 'megatron.core.models.gpt.bailing_moe_linear_v2_layer_specs',
                  'bailing_moe_linear_v2_block_spec',
        '--num-layers', '20',
        '--hidden-size', '2048',
        '--num-attention-heads', '16',
        '--num-query-groups', '16',
        '--ffn-hidden-size', '5120',
        '--group-query-attention',
        '--qk-layernorm',
        '--use-flash-attn',
        '--attention-dropout', '0',
        '--hidden-dropout', '0',
        '--position-embedding-type', 'rope',
        '--rotary-base', '10000',
        '--rotary-percent', '0.5',
        '--rotary-interleaved',
        '--max-position-embeddings', '4096',
        '--normalization', 'RMSNorm',
        '--norm-epsilon', '1e-06',
        '--swiglu',
        '--vocab-size', '157184',
        '--make-vocab-size-divisible-by', '128',
        '--untie-embeddings-and-output-weights',
        '--q-lora-rank', '256',
        '--kv-lora-rank', '512',
        '--qk-head-dim', '128',
        '--qk-pos-emb-head-dim', '64',
        '--v-head-dim', '128',
        # Linear attention
        '--linear-attention-freq', '([1]*4+[0]*1)*4',
        '--linear-attn-norm-group-size', '4',
        '--linear-attn-norm-group-type', 'group_diff',
        # MoE
        '--num-experts', '256',
        '--expert-model-parallel-size', '1',
        '--moe-router-topk', '8',
        '--moe-router-num-groups', '8',
        '--moe-router-group-topk', '4',
        '--moe-router-topk-scaling-factor', '2.5',
        '--moe-router-score-function', 'sigmoid',
        '--moe-router-enable-expert-bias',
        '--moe-router-bias-update-rate', '1e-3',
        '--moe-router-bias-zero-mean-update',
        '--moe-router-dtype', 'fp32',
        '--moe-token-dispatcher-type', 'alltoall',
        '--moe-grouped-gemm',
        '--moe-z-loss-coeff', '0.0000035',
        '--moe-ffn-hidden-size', '512',
        '--moe-shared-expert-intermediate-size', '2048',
        '--moe-layer-freq', '([0]+[1]*19)',
        # MTP
        '--mtp-num-layers', '1',
        '--mtp-loss-scaling-factor', '0.1',
        '--mtp-loss-scaling-per-layer',
        # Parallelism (all 1 for single-file)
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        # Minimal training args for model construction
        '--seq-length', '4096',
        '--train-iters', '95367',
        '--bf16',
        '--seed', '42',
        '--global-batch-size', '2',
        '--tokenizer-type', 'NullTokenizer',
        '--attention-backend', 'auto',
        '--attention-softmax-in-fp32',
        '--no-gradient-accumulation-fusion',
    ]

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables
    from megatron.core import mpu
    from megatron.legacy import fused_kernels
    from megatron.legacy.model import module
    from tools.checkpoint.utils import _ConverterFakeProcessGroup

    margs = parse_args()
    # Trick validate_args into thinking we have enough ranks
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    margs = validate_args(margs)

    # padded_vocab_size is normally set during tokenizer build (which we skip).
    # Compute it manually: ceil(vocab_size / (divisible_by * TP)) * (divisible_by * TP)
    import math
    multiple = margs.make_vocab_size_divisible_by * margs.tensor_model_parallel_size
    margs.padded_vocab_size = int(math.ceil(margs.vocab_size / multiple) * multiple)

    set_global_variables(margs, build_tokenizer=False)

    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)

    # Fake ALL process group globals so any module (including MTP's inner
    # TransformerLayer) can call ProcessGroupCollection.use_mpu_process_groups().
    fake_group = _ConverterFakeProcessGroup(size=1)
    for var_name in dir(mpu):
        if (var_name.startswith('_')
                and 'GROUP' in var_name
                and not var_name.endswith(('WORLD_SIZE', 'RANK', 'RANKS'))):
            if getattr(mpu, var_name) is None:
                if 'HIERARCHICAL' in var_name:
                    setattr(mpu, var_name, [fake_group])
                else:
                    setattr(mpu, var_name, fake_group)

    fused_kernels.load(margs)

    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)

    module.MegatronModule.embedding_warning_printed = True

    return margs


def build_model(margs):
    """Build the Megatron model."""
    from dataclasses import fields as dataclass_fields
    from model_provider import model_provider
    from gpt_builders import gpt_builder
    from megatron.core.process_groups_config import ProcessGroupCollection
    from tools.checkpoint.utils import _ConverterFakeProcessGroup

    # Construct a ProcessGroupCollection with all fields set to fake groups,
    # bypassing ProcessGroupCollection.use_mpu_process_groups() which requires
    # a fully initialized distributed environment.
    fake_group = _ConverterFakeProcessGroup(size=1)
    pg_collection = ProcessGroupCollection.__new__(ProcessGroupCollection)
    for f in dataclass_fields(ProcessGroupCollection):
        if f.name == 'hcp':
            object.__setattr__(pg_collection, f.name, [fake_group])
        else:
            object.__setattr__(pg_collection, f.name, fake_group)

    print("Building Megatron model...")
    model = model_provider(
        gpt_builder, pre_process=True, post_process=True, pg_collection=pg_collection
    ).to(margs.params_dtype)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model built with {num_params:,} parameters")
    return model


def _copy(dst, src, name=""):
    """Copy tensor data, asserting shape match."""
    assert dst.shape == src.shape, (
        f"Shape mismatch for {name}: model={dst.shape}, HF={src.shape}"
    )
    dst.data.copy_(src)


# ---------------------------------------------------------------------------
# Per-component weight copy helpers
# ---------------------------------------------------------------------------

def copy_linear_attn_weights(layer, hf_sd, hf_prefix):
    """Copy weights for a Linear Attention layer."""
    attn = layer.self_attention
    p = f"model.{hf_prefix}.attention"

    _copy(attn.linear_qkv.weight, hf_sd[f"{p}.query_key_value.weight"],
          f"{hf_prefix} linear_qkv")
    _copy(attn.linear_proj.weight, hf_sd[f"{p}.dense.weight"],
          f"{hf_prefix} linear_proj")
    _copy(attn.q_layernorm.weight, hf_sd[f"{p}.query_layernorm.weight"],
          f"{hf_prefix} q_layernorm")
    _copy(attn.k_layernorm.weight, hf_sd[f"{p}.key_layernorm.weight"],
          f"{hf_prefix} k_layernorm")
    _copy(attn.g_proj.weight, hf_sd[f"{p}.g_proj.weight"],
          f"{hf_prefix} g_proj")
    _copy(attn.g_norm.weight, hf_sd[f"{p}.g_norm.weight"],
          f"{hf_prefix} g_norm")


def copy_mla_weights(attn, hf_sd, hf_prefix):
    """Copy weights for an MLA (Multi-Latent Attention) layer.

    With TE backend + qk_layernorm, the q/kv layernorm weights are fused into
    the up_proj modules as layer_norm_weight (TELayerNormColumnParallelLinear).
    """
    p = f"model.{hf_prefix}.attention"

    _copy(attn.linear_q_down_proj.weight, hf_sd[f"{p}.q_a_proj.weight"],
          f"{hf_prefix} q_down_proj")
    _copy(attn.linear_q_up_proj.layer_norm_weight, hf_sd[f"{p}.q_a_layernorm.weight"],
          f"{hf_prefix} q_up_proj.layer_norm_weight")
    _copy(attn.linear_q_up_proj.weight, hf_sd[f"{p}.q_b_proj.weight"],
          f"{hf_prefix} q_up_proj")
    _copy(attn.linear_kv_down_proj.weight, hf_sd[f"{p}.kv_a_proj_with_mqa.weight"],
          f"{hf_prefix} kv_down_proj")
    _copy(attn.linear_kv_up_proj.layer_norm_weight, hf_sd[f"{p}.kv_a_layernorm.weight"],
          f"{hf_prefix} kv_up_proj.layer_norm_weight")
    _copy(attn.linear_kv_up_proj.weight, hf_sd[f"{p}.kv_b_proj.weight"],
          f"{hf_prefix} kv_up_proj")
    _copy(attn.linear_proj.weight, hf_sd[f"{p}.dense.weight"],
          f"{hf_prefix} linear_proj")


def copy_dense_mlp_weights(mlp, hf_sd, hf_prefix):
    """Copy weights for a dense MLP layer (SwiGLU: gate_proj + up_proj fused)."""
    p = f"model.{hf_prefix}.mlp"

    gate = hf_sd[f"{p}.gate_proj.weight"]
    up = hf_sd[f"{p}.up_proj.weight"]
    fused = torch.cat([gate, up], dim=0)
    _copy(mlp.linear_fc1.weight, fused, f"{hf_prefix} mlp.fc1")
    _copy(mlp.linear_fc2.weight, hf_sd[f"{p}.down_proj.weight"],
          f"{hf_prefix} mlp.fc2")


def copy_moe_weights(moe, hf_sd, hf_prefix, num_experts):
    """Copy weights for a MoE layer (router + experts + shared experts)."""
    p = f"model.{hf_prefix}.mlp"

    # Router weight
    _copy(moe.router.weight, hf_sd[f"{p}.gate.weight"],
          f"{hf_prefix} router")

    # Router expert_bias (buffer)
    expert_bias_key = f"{p}.gate.expert_bias"
    if expert_bias_key in hf_sd:
        moe.router.expert_bias.data.copy_(hf_sd[expert_bias_key])

    # Per-expert weights (TE GroupedLinear: weight0, weight1, ...)
    for e in range(num_experts):
        gate = hf_sd[f"{p}.experts.{e}.gate_proj.weight"]
        up = hf_sd[f"{p}.experts.{e}.up_proj.weight"]
        fused = torch.cat([gate, up], dim=0)
        fc1_w = getattr(moe.experts.linear_fc1, f"weight{e}")
        _copy(fc1_w, fused, f"{hf_prefix} expert{e}.fc1")

        fc2_w = getattr(moe.experts.linear_fc2, f"weight{e}")
        _copy(fc2_w, hf_sd[f"{p}.experts.{e}.down_proj.weight"],
              f"{hf_prefix} expert{e}.fc2")

    # Shared experts (SwiGLU fused)
    gate = hf_sd[f"{p}.shared_experts.gate_proj.weight"]
    up = hf_sd[f"{p}.shared_experts.up_proj.weight"]
    fused = torch.cat([gate, up], dim=0)
    _copy(moe.shared_experts.linear_fc1.weight, fused,
          f"{hf_prefix} shared_experts.fc1")
    _copy(moe.shared_experts.linear_fc2.weight,
          hf_sd[f"{p}.shared_experts.down_proj.weight"],
          f"{hf_prefix} shared_experts.fc2")


# ---------------------------------------------------------------------------
# Main weight copy orchestration
# ---------------------------------------------------------------------------

def copy_weights(model, hf_sd, margs):
    """Copy all weights from HF state dict to Megatron model."""
    num_layers = margs.num_layers
    num_experts = margs.num_experts

    # Compute per-layer patterns (may already be lists after validate_args)
    la_freq = margs.linear_attention_freq
    la_pattern = eval(la_freq) if isinstance(la_freq, str) else la_freq  # noqa: S307
    moe_freq = margs.moe_layer_freq
    moe_pattern = eval(moe_freq) if isinstance(moe_freq, str) else moe_freq  # noqa: S307

    # Embeddings
    print("Copying embeddings...")
    _copy(model.embedding.word_embeddings.weight,
          hf_sd['model.word_embeddings.weight'], 'word_embeddings')

    # Decoder layers
    print("Copying decoder layers...")
    for i in range(num_layers):
        layer = model.decoder.layers[i]
        hf_pfx = f"layers.{i}"

        # Layernorms
        _copy(layer.input_layernorm.weight,
              hf_sd[f"model.{hf_pfx}.input_layernorm.weight"],
              f"layer {i} input_layernorm")
        _copy(layer.pre_mlp_layernorm.weight,
              hf_sd[f"model.{hf_pfx}.post_attention_layernorm.weight"],
              f"layer {i} pre_mlp_layernorm")

        # Attention
        if la_pattern[i]:
            copy_linear_attn_weights(layer, hf_sd, hf_pfx)
        else:
            copy_mla_weights(layer.self_attention, hf_sd, hf_pfx)

        # MLP
        if moe_pattern[i]:
            copy_moe_weights(layer.mlp, hf_sd, hf_pfx, num_experts)
        else:
            copy_dense_mlp_weights(layer.mlp, hf_sd, hf_pfx)

        la_type = 'LinearAttn' if la_pattern[i] else 'MLA'
        mlp_type = 'MoE' if moe_pattern[i] else 'Dense'
        print(f"  Layer {i:2d} done ({la_type} + {mlp_type})")

    # Final layernorm
    print("Copying final layernorm...")
    _copy(model.decoder.final_layernorm.weight,
          hf_sd['model.norm.weight'], 'final_layernorm')

    # Output layer
    print("Copying output layer...")
    _copy(model.output_layer.weight,
          hf_sd['lm_head.weight'], 'output_layer')

    # MTP layer (HF layer 20 -> Megatron mtp.layers.0)
    if hasattr(model, 'mtp') and model.mtp is not None:
        print("Copying MTP layer...")
        mtp_layer = model.mtp.layers[0]
        hf_mtp = f"layers.{num_layers}"

        _copy(mtp_layer.enorm.weight,
              hf_sd[f"model.{hf_mtp}.enorm.weight"], 'mtp enorm')
        _copy(mtp_layer.hnorm.weight,
              hf_sd[f"model.{hf_mtp}.hnorm.weight"], 'mtp hnorm')
        _copy(mtp_layer.eh_proj.weight,
              hf_sd[f"model.{hf_mtp}.eh_proj.weight"], 'mtp eh_proj')
        _copy(mtp_layer.final_layernorm.weight,
              hf_sd[f"model.{hf_mtp}.final_layernorm.weight"],
              'mtp final_layernorm')

        # MTP inner transformer layer (MLA + MoE, same as last decoder layer)
        inner = mtp_layer.mtp_model_layer

        _copy(inner.input_layernorm.weight,
              hf_sd[f"model.{hf_mtp}.input_layernorm.weight"],
              'mtp inner input_layernorm')
        _copy(inner.pre_mlp_layernorm.weight,
              hf_sd[f"model.{hf_mtp}.post_attention_layernorm.weight"],
              'mtp inner pre_mlp_layernorm')

        copy_mla_weights(inner.self_attention, hf_sd, hf_mtp)
        copy_moe_weights(inner.mlp, hf_sd, hf_mtp, num_experts)
        print("  MTP layer done")

    print("All weights copied successfully!")


def _shard_state_dict_for_ep(full_sd, ep_size, num_experts):
    """Split a flat EP=1 state dict into EP-sharded state dicts.

    Expert weight keys match ``*.experts.linear_fc{1,2}.weight{N}`` where N
    is the global expert index (0..num_experts-1).  Each EP shard keeps only
    its local experts and renumbers them 0..num_local_experts-1.

    Non-expert keys (layernorms, attention, router, shared_experts, etc.) are
    duplicated across all shards.
    """
    import re

    assert ep_size > 0, f"ep_size must be positive, got {ep_size}"
    assert num_experts % ep_size == 0, (
        f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
    )

    expert_weight_re = re.compile(
        r'^(.*\.experts\.linear_fc[12])\.weight(\d+)$'
    )
    num_local = num_experts // ep_size

    shards = [{} for _ in range(ep_size)]
    for key, tensor in full_sd.items():
        m = expert_weight_re.match(key)
        if m:
            prefix = m.group(1)
            global_idx = int(m.group(2))
            ep_rank = global_idx // num_local
            local_idx = global_idx % num_local
            shards[ep_rank][f"{prefix}.weight{local_idx}"] = tensor
        else:
            # Non-expert params: duplicate to all shards
            for shard in shards:
                shard[key] = tensor

    for rank, shard in enumerate(shards):
        n_expert_keys = sum(1 for k in shard if expert_weight_re.match(k))
        expected = num_local * 2  # fc1 + fc2 per expert (per MoE layer counted separately)
        print(f"  EP shard {rank}: {len(shard)} keys, {n_expert_keys} expert weight keys")

    return shards


def save_checkpoint(model, output_dir, iteration, margs, target_ep_size=1):
    """Save model in Megatron legacy torch format.

    When *target_ep_size* > 1 the checkpoint is sharded across expert-parallel
    ranks, producing ``mp_rank_00_{ep_rank:03d}/model_optim_rng.pt`` for each
    rank (matching what ``get_checkpoint_name`` generates with EP enabled).
    """
    full_sd = model.state_dict_for_save_checkpoint()

    if target_ep_size > 1:
        print(f"Sharding checkpoint for EP={target_ep_size} "
              f"({margs.num_experts} experts → "
              f"{margs.num_experts // target_ep_size} per shard)...")
        shards = _shard_state_dict_for_ep(
            full_sd, target_ep_size, margs.num_experts,
        )
    else:
        shards = [full_sd]

    for ep_rank, shard_sd in enumerate(shards):
        if target_ep_size > 1:
            rank_dir = os.path.join(
                output_dir, f'iter_{iteration:07d}', f'mp_rank_00_{ep_rank:03d}',
            )
        else:
            rank_dir = os.path.join(
                output_dir, f'iter_{iteration:07d}', 'mp_rank_00',
            )
        os.makedirs(rank_dir, exist_ok=True)

        ckpt = {
            'iteration': iteration,
            'model': shard_sd,
            'args': margs,
            'checkpoint_version': 3.0,
        }
        ckpt_path = os.path.join(rank_dir, 'model_optim_rng.pt')
        print(f"Saving EP shard {ep_rank} to {ckpt_path}...")
        torch.save(ckpt, ckpt_path)

    tracker_path = os.path.join(output_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_path, 'w') as f:
        f.write(str(iteration))

    print(f"Checkpoint saved ({len(shards)} shard(s)). "
          f"Tracker points to iteration {iteration}.")


def main():
    args = parse_args()

    hf_sd = load_hf_state_dict(args.input_dir)
    margs = initialize_megatron()
    model = build_model(margs)
    copy_weights(model, hf_sd, margs)
    save_checkpoint(
        model, args.output_dir, args.iteration, margs,
        target_ep_size=args.target_expert_parallel_size,
    )

    print("Conversion complete!")


if __name__ == '__main__':
    main()
