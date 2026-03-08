# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer specifications for BailingMoE Linear V2 with custom MTP layer."""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.linear_attention import LinearAttention
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


def get_bailing_moe_v2_mtp_layer_submodules(
    backend: BackendSpecProvider, mtp_model_layer_spec: ModuleSpec
) -> MultiTokenPredictionLayerSubmodules:
    """
    Get MTP layer submodules for BailingMoE Linear V2.

    This is a custom implementation that adds final_layernorm to match the
    reference BailingMoE Linear V2 implementation.

    Args:
        backend: Backend specification provider (TE or Local)
        mtp_model_layer_spec: Specification for the transformer layer

    Returns:
        MultiTokenPredictionLayerSubmodules with final_layernorm added
    """
    layer_norm_impl = backend.layer_norm()

    return MultiTokenPredictionLayerSubmodules(
        enorm=layer_norm_impl,
        hnorm=layer_norm_impl,
        eh_proj=backend.column_parallel_linear(),
        mtp_model_layer=mtp_model_layer_spec,
        layer_norm=layer_norm_impl,  # This is the final_layernorm
    )


def get_bailing_moe_v2_mtp_layer_spec(
    backend: BackendSpecProvider, mtp_model_layer_spec: ModuleSpec
) -> ModuleSpec:
    """
    Get MTP layer spec for BailingMoE Linear V2.

    Args:
        backend: Backend specification provider (TE or Local)
        mtp_model_layer_spec: Specification for the transformer layer

    Returns:
        ModuleSpec for MultiTokenPredictionLayer with custom submodules
    """
    return ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=get_bailing_moe_v2_mtp_layer_submodules(backend, mtp_model_layer_spec),
    )


def get_bailing_moe_linear_v2_layer_spec(
    backend: BackendSpecProvider, layer_idx: int, config: TransformerConfig
) -> TransformerLayerSubmodules:
    """
    Get layer spec for BailingMoE Linear V2 with hybrid architecture.

    This function determines the layer type based on layer_idx:
    - Attention: Linear Attention for layers 0-3, 5-8, 10-13, 15-18
                 Standard Attention for layers 4, 9, 14, 19
    - MLP: Dense MLP for layer 0, MoE for layers 1-19

    Args:
        backend: Backend specification provider (TE or Local)
        layer_idx: Layer index (0-based)
        config: Transformer configuration

    Returns:
        TransformerLayerSubmodules with appropriate specs for this layer
    """
    # Determine attention type based on layer_group_size=5 pattern
    # Layers 0-3, 5-8, 10-13, 15-18: Linear Attention
    # Layers 4, 9, 14, 19: Standard Attention
    use_linear_attention = (layer_idx + 1) % 5 != 0

    # Determine MLP type
    # Layer 0: Dense MLP
    # Layers 1-19: MoE
    use_moe = layer_idx >= 1

    # Get layer norm
    layer_norm = backend.layer_norm()
    # Get QK layer norm (use specialized builder for QK norms)
    qk_layer_norm = backend.layer_norm(for_qk=True) if config.qk_layernorm else IdentityOp

    # Build attention spec
    if use_linear_attention:
        # Linear Attention
        attention_spec = ModuleSpec(
            module=LinearAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=qk_layer_norm,
                k_layernorm=qk_layer_norm,
            ),
        )
    else:
        # Standard Attention
        attention_spec = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=qk_layer_norm,
                k_layernorm=qk_layer_norm,
            ),
        )

    # Build MLP spec
    if use_moe:
        # MoE
        mlp_spec = get_moe_module_spec_for_backend(
            backend=backend,
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        )
    else:
        # Dense MLP
        mlp_spec = get_mlp_module_spec_for_backend(backend=backend)

    # Return TransformerLayerSubmodules
    return TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=attention_spec,
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )


def bailing_moe_linear_v2_block_spec(
    config: TransformerConfig,
) -> TransformerBlockSubmodules:
    """
    Block spec factory for BailingMoE Linear V2, referenced via --spec argument.

    Called by gpt_builders.py with config when --spec points to this function.
    Follows the same pattern as get_gpt_decoder_block_spec: builds per-layer specs,
    slices for pipeline parallelism, and returns TransformerBlockSubmodules.

    Args:
        config: Transformer configuration

    Returns:
        TransformerBlockSubmodules with BailingMoE Linear V2 layers
    """
    from megatron.core.models.backends import TESpecProvider, LocalSpecProvider
    from megatron.core.transformer.custom_layers.transformer_engine import TENorm
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm as LNImpl
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    use_te = config.transformer_impl == "transformer_engine"

    # Get backend
    if use_te:
        backend = TESpecProvider()
    else:
        backend = LocalSpecProvider()

    # Build layer specs list for all layers
    layer_specs = [
        ModuleSpec(
            module=TransformerLayer,
            submodules=get_bailing_moe_linear_v2_layer_spec(backend, layer_idx, config),
        )
        for layer_idx in range(config.num_layers)
    ]

    # Slice layer specs for pipeline parallelism (following get_gpt_decoder_block_spec)
    num_layers_to_build = get_num_layers_to_build(config)
    offset = get_transformer_layer_offset(config)
    local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Determine layer norm implementation
    if use_te:
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl

    return TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=layer_norm_impl,
    )
