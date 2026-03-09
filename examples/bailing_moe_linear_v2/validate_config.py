#!/usr/bin/env python3
"""
Validation script for BailingMoE Linear V2 training configuration.

This script validates that the training configuration in pretrain_bailing_moe_linear_v2.sh
matches the reference implementation parameters from ref/ling2.5-pretrain/bailing_moe_modeling.py.

Usage:
    python examples/bailing_moe_linear_v2/validate_config.py

Exit codes:
    0: All parameters match
    1: Parameter mismatch detected
    2: Script parsing error
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Reference implementation parameters from ref/ling2.5-pretrain/bailing_moe_modeling.py
REFERENCE_PARAMS = {
    # Model Architecture
    "num_layers": 20,
    "hidden_size": 2048,
    "ffn_hidden_size": 5120,  # Dense MLP hidden size for layer 0
    "num_attention_heads": 16,
    "num_key_value_heads": 4,

    # MoE Configuration
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 1,
    "moe_aux_loss_coeff": 0.0000035,
    "score_function": "sigmoid",
    "enable_expert_bias": True,

    # Linear Attention Configuration
    "layer_group_size": 5,  # 4 linear + 1 standard
    "group_norm_size": 4,

    # Attention Configuration
    "use_qk_norm": True,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,

    # Position Embeddings
    "partial_rotary_factor": 0.5,
    "rope_theta": 10000,
    "max_position_embeddings": 4096,

    # Vocabulary
    "vocab_size": 157184,

    # MLA Dimension Configuration
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_head_dim": 128,
    "qk_pos_emb_head_dim": 64,
    "v_head_dim": 128,

    # Position Embedding Configuration
    "rotary_interleaved": True,

    # MTP Configuration
    "num_nextn_predict_layers": 1,
    "mtp_loss_scaling_factor": 0.1,
    "mtp_loss_scaling_per_layer": True,

    # Training Configuration
    "micro_batch_size": 2,
    "global_batch_size": 5120,
    "seq_length": 4096,
    "train_iters": 95367,
    "lr": 0.000336,
    "min_lr": 0.000336,
    "lr_warmup_iters": 2000,
    "weight_decay": 0.1,
    "clip_grad": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "init_method_std": 0.006,
}


def extract_script_params(script_path: Path) -> Dict[str, Any]:
    """
    Extract configuration parameters from the training script.

    Args:
        script_path: Path to pretrain_bailing_moe_linear_v2.sh

    Returns:
        Dictionary of extracted parameters
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    with open(script_path, 'r') as f:
        content = f.read()

    params = {}

    # Extract parameters using regex patterns
    patterns = {
        # Model Architecture
        "num_layers": r"--num-layers\s+(\d+)",
        "hidden_size": r"--hidden-size\s+(\d+)",
        "num_attention_heads": r"--num-attention-heads\s+(\d+)",
        "num_key_value_heads": r"--num-query-groups\s+(\d+)",
        "ffn_hidden_size": r"--ffn-hidden-size\s+(\d+)",

        # MoE Configuration
        "num_experts": r"--num-experts\s+(\d+)",
        "moe_router_topk": r"--moe-router-topk\s+(\d+)",
        "moe_router_num_groups": r"--moe-router-num-groups\s+(\d+)",
        "moe_router_group_topk": r"--moe-router-group-topk\s+(\d+)",
        "moe_router_topk_scaling_factor": r"--moe-router-topk-scaling-factor\s+([\d.]+)",
        "moe_router_score_function": r"--moe-router-score-function\s+(\w+)",
        "moe_ffn_hidden_size": r"--moe-ffn-hidden-size\s+(\d+)",
        "moe_shared_expert_intermediate_size": r"--moe-shared-expert-intermediate-size\s+(\d+)",
        "moe_aux_loss_coeff": r"--moe-aux-loss-coeff\s+([\d.eE+-]+)",
        "moe_layer_freq": r'--moe-layer-freq\s+"([^"]+)"',

        # Linear Attention
        "linear_attention_freq": r'--linear-attention-freq\s+"([^"]+)"',
        "linear_attn_norm_group_size": r"--linear-attn-norm-group-size\s+(\d+)",

        # Attention Configuration
        "attention_dropout": r"--attention-dropout\s+([\d.]+)",
        "hidden_dropout": r"--hidden-dropout\s+([\d.]+)",

        # Position Embeddings
        "rotary_percent": r"--rotary-percent\s+([\d.]+)",
        "rotary_base": r"--rotary-base\s+(\d+)",
        "max_position_embeddings": r"--max-position-embeddings\s+(\d+)",

        # Vocabulary
        "vocab_size": r"--vocab-size\s+(\d+)",

        # MLA Dimension Configuration
        "q_lora_rank": r"--q-lora-rank\s+(\d+)",
        "kv_lora_rank": r"--kv-lora-rank\s+(\d+)",
        "qk_head_dim": r"--qk-head-dim\s+(\d+)",
        "qk_pos_emb_head_dim": r"--qk-pos-emb-head-dim\s+(\d+)",
        "v_head_dim": r"--v-head-dim\s+(\d+)",

        # MTP Configuration
        "mtp_num_layers": r"--mtp-num-layers\s+(\d+)",
        "mtp_loss_scaling_factor": r"--mtp-loss-scaling-factor\s+([\d.]+)",

        # Training Configuration
        "micro_batch_size": r"--micro-batch-size\s+(\d+)",
        "global_batch_size": r"--global-batch-size\s+(\d+)",
        "seq_length": r"--seq-length\s+(\d+)",
        "train_iters": r"--train-iters\s+(\d+)",
        "lr": r"--lr\s+([\d.]+)",
        "min_lr": r"--min-lr\s+([\d.]+)",
        "lr_warmup_iters": r"--lr-warmup-iters\s+(\d+)",
        "weight_decay": r"--weight-decay\s+([\d.]+)",
        "clip_grad": r"--clip-grad\s+([\d.]+)",
        "adam_beta1": r"--adam-beta1\s+([\d.]+)",
        "adam_beta2": r"--adam-beta2\s+([\d.]+)",
        "init_method_std": r"--init-method-std\s+([\d.]+)",
    }

    for param_name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if param_name in ["moe_router_score_function", "linear_attention_freq", "moe_layer_freq"]:
                params[param_name] = value
            elif '.' in value or 'e' in value.lower():
                params[param_name] = float(value)
            else:
                params[param_name] = int(value)

    # Check for boolean flags
    params["qk_layernorm"] = "--qk-layernorm" in content
    params["moe_router_enable_expert_bias"] = "--moe-router-enable-expert-bias" in content
    params["rotary_interleaved"] = "--rotary-interleaved" in content
    params["mtp_loss_scaling_per_layer"] = "--mtp-loss-scaling-per-layer" in content

    return params


def validate_parameters(script_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate script parameters against reference implementation.

    Args:
        script_params: Parameters extracted from training script

    Returns:
        Tuple of (all_valid, error_messages)
    """
    errors = []
    all_valid = True

    # Mapping from script parameter names to reference parameter names
    param_mapping = {
        # Model Architecture
        "num_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "ffn_hidden_size": "ffn_hidden_size",

        # MoE Configuration
        "num_experts": "num_experts",
        "moe_router_topk": "num_experts_per_tok",
        "moe_router_num_groups": "n_group",
        "moe_router_group_topk": "topk_group",
        "moe_router_topk_scaling_factor": "routed_scaling_factor",
        "moe_router_score_function": "score_function",
        "moe_ffn_hidden_size": "moe_intermediate_size",
        "moe_shared_expert_intermediate_size": "shared_expert_intermediate_size",
        "moe_aux_loss_coeff": "moe_aux_loss_coeff",
        "moe_router_enable_expert_bias": "enable_expert_bias",

        # Linear Attention
        "linear_attn_norm_group_size": "group_norm_size",

        # Attention Configuration
        "qk_layernorm": "use_qk_norm",
        "attention_dropout": "attention_dropout",
        "hidden_dropout": "hidden_dropout",

        # Position Embeddings
        "rotary_percent": "partial_rotary_factor",
        "rotary_base": "rope_theta",
        "max_position_embeddings": "max_position_embeddings",

        # Vocabulary
        "vocab_size": "vocab_size",

        # MLA Dimension Configuration
        "q_lora_rank": "q_lora_rank",
        "kv_lora_rank": "kv_lora_rank",
        "qk_head_dim": "qk_head_dim",
        "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
        "v_head_dim": "v_head_dim",

        # Position Embedding Configuration
        "rotary_interleaved": "rotary_interleaved",

        # MTP Configuration
        "mtp_num_layers": "num_nextn_predict_layers",
        "mtp_loss_scaling_factor": "mtp_loss_scaling_factor",
        "mtp_loss_scaling_per_layer": "mtp_loss_scaling_per_layer",

        # Training Configuration
        "micro_batch_size": "micro_batch_size",
        "global_batch_size": "global_batch_size",
        "seq_length": "seq_length",
        "train_iters": "train_iters",
        "lr": "lr",
        "min_lr": "min_lr",
        "lr_warmup_iters": "lr_warmup_iters",
        "weight_decay": "weight_decay",
        "clip_grad": "clip_grad",
        "adam_beta1": "adam_beta1",
        "adam_beta2": "adam_beta2",
        "init_method_std": "init_method_std",
    }

    # Validate each parameter
    for script_param, ref_param in param_mapping.items():
        if ref_param is None:
            continue

        if script_param not in script_params:
            errors.append(f"❌ Missing parameter in script: {script_param}")
            all_valid = False
            continue

        script_value = script_params[script_param]
        ref_value = REFERENCE_PARAMS[ref_param]

        # Compare values
        if isinstance(ref_value, float):
            # Use relative tolerance for floating point comparison
            if abs(script_value - ref_value) > 1e-9:
                errors.append(
                    f"❌ {script_param}: script={script_value}, reference={ref_value}"
                )
                all_valid = False
        else:
            if script_value != ref_value:
                errors.append(
                    f"❌ {script_param}: script={script_value}, reference={ref_value}"
                )
                all_valid = False

    # Special validation: linear attention frequency pattern
    if "linear_attention_freq" in script_params:
        freq_pattern = script_params["linear_attention_freq"]
        # Pattern should be "([1]*4+[0]*1)*4" which means 4 linear + 1 standard, repeated 4 times
        if freq_pattern != "([1]*4+[0]*1)*4":
            errors.append(
                f"❌ linear_attention_freq: script={freq_pattern}, expected=([1]*4+[0]*1)*4"
            )
            all_valid = False
    else:
        errors.append("❌ Missing parameter in script: linear_attention_freq")
        all_valid = False

    # Special validation: MoE layer frequency pattern
    if "moe_layer_freq" in script_params:
        moe_freq = script_params["moe_layer_freq"]
        # Pattern should be "([0]+[1]*19)" which means first layer dense, rest MoE
        if moe_freq != "([0]+[1]*19)":
            errors.append(
                f"❌ moe_layer_freq: script={moe_freq}, expected=([0]+[1]*19)"
            )
            all_valid = False
    else:
        errors.append("❌ Missing parameter in script: moe_layer_freq")
        all_valid = False

    return all_valid, errors


def print_validation_report(all_valid: bool, errors: List[str], script_params: Dict[str, Any]):
    """
    Print a detailed validation report.

    Args:
        all_valid: Whether all parameters are valid
        errors: List of error messages
        script_params: Parameters extracted from script
    """
    print("=" * 80)
    print("BailingMoE Linear V2 Configuration Validation Report")
    print("=" * 80)
    print()

    if all_valid:
        print("✅ SUCCESS: All parameters match the reference implementation!")
        print()
        print("Key Configuration Summary:")
        print(f"  - Model: {script_params.get('num_layers', 'N/A')} layers, "
              f"{script_params.get('hidden_size', 'N/A')} hidden size")
        print(f"  - MoE: {script_params.get('num_experts', 'N/A')} experts, "
              f"TopK={script_params.get('moe_router_topk', 'N/A')}")
        print(f"  - Linear Attention: {script_params.get('linear_attention_freq', 'N/A')}")
        print(f"  - Training: {script_params.get('train_iters', 'N/A')} iterations, "
              f"LR={script_params.get('lr', 'N/A')}")
    else:
        print("❌ VALIDATION FAILED: Parameter mismatches detected!")
        print()
        print("Errors:")
        for error in errors:
            print(f"  {error}")
        print()
        print(f"Total errors: {len(errors)}")

    print()
    print("=" * 80)


def main():
    """Main validation function."""
    # Find the training script
    script_path = Path(__file__).parent.parent.parent / "pretrain_bailing_moe_linear_v2.sh"

    if not script_path.exists():
        print(f"❌ ERROR: Training script not found at {script_path}")
        print("Please ensure pretrain_bailing_moe_linear_v2.sh exists in the repository root.")
        sys.exit(2)

    try:
        # Extract parameters from script
        print("Extracting parameters from training script...")
        script_params = extract_script_params(script_path)
        print(f"✓ Extracted {len(script_params)} parameters")
        print()

        # Validate parameters
        print("Validating parameters against reference implementation...")
        all_valid, errors = validate_parameters(script_params)
        print()

        # Print report
        print_validation_report(all_valid, errors, script_params)

        # Exit with appropriate code
        sys.exit(0 if all_valid else 1)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
