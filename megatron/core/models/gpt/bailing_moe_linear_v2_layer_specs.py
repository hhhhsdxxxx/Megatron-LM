# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer specifications for BailingMoE Linear V2 hybrid architecture."""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
    get_moe_layer_pattern,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.linear_attention import LinearAttention
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


def get_bailing_moe_linear_v2_layer_spec(
    backend: BackendSpecProvider,
    config: TransformerConfig,
    use_linear_attention: bool,
    use_moe: bool,
    use_te: bool,
) -> TransformerLayerSubmodules:
    """
    Get layer spec for BailingMoE Linear V2 with hybrid architecture.

    Attention and MLP types are determined by the caller based on
    config.linear_attention_freq and config.moe_layer_freq patterns.

    Args:
        backend: Backend specification provider (TE or Local)
        config: Transformer configuration
        use_linear_attention: Whether this layer uses Linear Attention (vs MLA)
        use_moe: Whether this layer uses MoE (vs Dense MLP)
        use_te: Whether to use Transformer Engine backend

    Returns:
        TransformerLayerSubmodules with appropriate specs for this layer
    """

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
                linear_g_proj=ColumnParallelLinear,
                q_layernorm=qk_layer_norm,
                k_layernorm=qk_layer_norm,
            ),
        )
    else:
        # MLA (Multi-Latent Attention)
        if use_te:
            # TE backend: down_proj uses non-parallel linear, layernorms fused into up_proj
            up_proj = (
                backend.column_parallel_layer_norm_linear()
                if config.qk_layernorm
                else backend.column_parallel_linear()
            )
            attention_spec = ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.linear(),
                    linear_q_up_proj=up_proj,
                    linear_kv_down_proj=backend.linear(),
                    linear_kv_up_proj=up_proj,
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                ),
            )
        else:
            # Local backend: all linears are column_parallel, explicit layernorms
            mla_layernorm = backend.layer_norm(for_qk=True) if config.qk_layernorm else IdentityOp
            attention_spec = ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.column_parallel_linear(),
                    linear_q_up_proj=backend.column_parallel_linear(),
                    linear_kv_down_proj=backend.column_parallel_linear(),
                    linear_kv_up_proj=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=mla_layernorm,
                    kv_layernorm=mla_layernorm,
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
    Builds per-layer specs with hybrid Linear Attention / MLA and Dense / MoE,
    slices for pipeline parallelism, and returns TransformerBlockSubmodules.

    MTP handling: gpt_builders.py passes this block spec to get_gpt_mtp_block_spec,
    which extracts the last decoder layer (MLA + MoE) as the MTP model layer.
    Note on MTP RoPE: The reference passes the linear-attention RoPE to MTP's MLA,
    but MLA ignores external rotary_pos_emb (asserts it's None) and uses its own
    internal RoPE with qk_pos_emb_head_dim. This is correct because both produce
    the same dimension (qk_rope_head_dim == head_dim * partial_rotary_factor).

    The layer architecture is determined by config parameters:
    - config.linear_attention_freq: Controls which layers use Linear Attention vs MLA
    - config.moe_layer_freq: Controls which layers use MoE vs Dense MLP

    Args:
        config: Transformer configuration

    Returns:
        TransformerBlockSubmodules with BailingMoE Linear V2 layers
    """
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm as LNImpl
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    use_te = config.transformer_impl == "transformer_engine"

    # Validate: reference BailingMoE v2.5 has no gate on shared experts
    assert not config.moe_shared_expert_gate, (
        "BailingMoE Linear V2 reference has no gate on shared experts. "
        "Set --no-moe-shared-expert-gate or remove --moe-shared-expert-gate."
    )

    # Validate: reference BailingMoE v2.5 uses the same bias setting for both QKV and
    # output projection in MLA. Megatron controls output projection bias via add_bias_linear
    # and QKV bias via add_qkv_bias. These must be consistent to align with the reference.
    if config.add_qkv_bias != config.add_bias_linear:
        import warnings

        warnings.warn(
            "BailingMoE Linear V2 reference uses the same bias setting (use_qkv_bias) for "
            "both QKV projections and the MLA output projection. In Megatron, add_qkv_bias="
            f"{config.add_qkv_bias} but add_bias_linear={config.add_bias_linear}. "
            "This mismatch will cause the MLA output projection bias to differ from the "
            "reference. Set both to the same value for alignment.",
            stacklevel=2,
        )

    # Get backend
    if use_te:
        backend = TESpecProvider()
    else:
        backend = LocalSpecProvider()

    # Compute per-layer patterns from config
    la_pattern = get_linear_attention_pattern(config)
    moe_pattern = get_moe_layer_pattern(config)

    # Validate MLA dimension fields are set when pattern includes MLA layers.
    # Note: q_lora_rank is intentionally excluded — MLASelfAttention supports
    # q_lora_rank=None (direct query projection without LoRA).
    if any(not la_pattern[i] for i in range(config.num_layers)):
        required_mla_fields = {
            "kv_lora_rank": "--kv-lora-rank",
            "qk_head_dim": "--qk-head-dim",
            "qk_pos_emb_head_dim": "--qk-pos-emb-head-dim",
            "v_head_dim": "--v-head-dim",
        }
        missing = [
            arg for field, arg in required_mla_fields.items() if getattr(config, field) is None
        ]
        assert not missing, (
            "MLA layers are present but required dimension fields are missing. "
            f"Please provide: {', '.join(missing)}"
        )

    # Build layer specs list for all layers
    layer_specs = [
        ModuleSpec(
            module=TransformerLayer,
            submodules=get_bailing_moe_linear_v2_layer_spec(
                backend,
                config,
                use_linear_attention=bool(la_pattern[layer_idx]),
                use_moe=bool(moe_pattern[layer_idx]),
                use_te=use_te,
            ),
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
