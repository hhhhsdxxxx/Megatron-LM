# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""LinearAttention (Gated Linear Attention) implementation for Megatron-LM.

This module implements the LinearAttention mechanism based on Lightning Attention-2,
which uses slope tensors to compute decay factors for efficient attention computation.
"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.group_rms_norm import GroupRMSNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

# Import GLA operators
try:
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
    HAVE_GLA = True
except ImportError:
    chunk_simple_gla = None
    fused_recurrent_simple_gla = None
    HAVE_GLA = False


class LinearAttention(Attention):
    """LinearAttention layer implementing Gated Linear Attention.

    This class extends the base Attention class to implement LinearAttention
    with slope-based decay factors as described in Lightning Attention-2.

    The slope tensor is used to compute position-dependent decay factors that
    enable efficient linear-time attention computation while maintaining
    competitive performance with standard attention mechanisms.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
        """Initialize LinearAttention layer.

        Args:
            config: Transformer configuration
            submodules: Attention submodules specification
            layer_number: Layer index in the model
            attn_mask_type: Type of attention mask (default: causal)
            attention_type: Type of attention (default: "self")
            cp_comm_type: Context parallel communication type
            pg_collection: Process group collection for distributed training
            pp_layer_offset: Pipeline parallel layer offset
        """
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

        # Check GLA operators availability
        if not HAVE_GLA:
            raise ImportError(
                "GLA operators not found. Please install fla package: "
                "pip install fla or from https://github.com/sustcsonglin/flash-linear-attention"
            )

        # Store GLA operators in dict
        self.gla_ops = {
            'chunk': chunk_simple_gla,
            'fused_recurrent': fused_recurrent_simple_gla,
        }

        # Build layer-dependent slope tensor
        # The slope varies by layer to provide different decay patterns
        # Formula: slope = -base_slope * (1 - (layer_idx - 1) / (num_layers - 1) + 1e-5)
        base_slope = self._build_slope_tensor(config.num_attention_heads)
        layer_idx = layer_number
        num_layers = config.num_layers
        slope = -base_slope * (1 - (layer_idx - 1) / max(num_layers - 1, 1) + 1e-5)

        # Register slope as buffer (non-persistent, will be recomputed)
        self.register_buffer('slope', slope, persistent=False)

        # Initialize GroupRMSNorm for gating
        # Note: hidden_size is set to num_heads * kv_channels to match the gating dimension
        # after g_proj projection, not the model's hidden_size
        self.g_norm = GroupRMSNorm(
            hidden_size=config.num_attention_heads * config.kv_channels,
            group_norm_size=config.linear_attn_norm_group_size,
            eps=config.layernorm_epsilon,
            sequence_parallel=config.sequence_parallel,
        )

        # Initialize gating projection
        # Projects from hidden_size to num_heads * kv_channels
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.kv_channels,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
        )

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int) -> Tensor:
        """Build slope tensor for Lightning Attention-2.

        Computes decay factors for each attention head based on the Lightning
        Attention-2 algorithm. The slopes form a geometric sequence that provides
        different decay rates for different heads.

        For power-of-2 head counts, uses the standard formula:
            start = 2 ** (-(2 ** -(log2(n) - 3)))
            slopes[i] = start * (start ** i)

        For non-power-of-2 head counts, uses a workaround that interpolates
        between the closest power-of-2 values.

        Args:
            n_attention_heads: Number of attention heads

        Returns:
            Tensor of shape [n_attention_heads] containing slope values
        """
        n = n_attention_heads

        def get_slopes_power_of_2(n: int) -> Tensor:
            """Get slopes for power-of-2 head counts."""
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)], dtype=torch.float32)

        # Check if n is a power of 2
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            # For non-power-of-2, use workaround from Lightning Attention
            # This concatenates slopes from two power-of-2 sequences
            closest_power_of_2 = 2 ** math.floor(math.log2(n))

            # Get slopes for the closest smaller power of 2
            slopes_1 = get_slopes_power_of_2(closest_power_of_2)

            # Get slopes for the next power of 2, taking every other element (even indices)
            slopes_2_all = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes_2 = slopes_2_all[0::2][:n - closest_power_of_2]

            # Concatenate
            slopes = torch.cat([slopes_1, slopes_2])

            return slopes

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor | None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, list[int]]:
        """
        Get query, key, value tensors for LinearAttention.

        This method implements the abstract method from the Attention base class.
        For LinearAttention, we project hidden_states to QKV and optionally split them.

        Args:
            hidden_states: Input tensor [sq, b, h]
            key_value_states: Key/value states for cross attention (not used in LinearAttention)
            output_gate: Whether to output gate tensor for gating mechanism
            split_qkv: Whether to split QKV into separate tensors

        Returns:
            If output_gate=True: (q, k, v, gate)
            If split_qkv=True: (q, k, v)
            If split_qkv=False: (qkv, split_indices)
        """
        # TODO: Full implementation in Phase 2
        # For now, return placeholder to satisfy abstract method requirement
        if not split_qkv:
            # Return packed QKV with split indices
            qkv, _ = self.linear_qkv(hidden_states)
            # Split indices for [q_heads, kv_heads, kv_heads]
            split_indices = [
                self.config.num_attention_heads * self.config.kv_channels,
                self.config.num_query_groups * self.config.kv_channels,
                self.config.num_query_groups * self.config.kv_channels,
            ]
            return qkv, split_indices

        # Project to QKV
        qkv, _ = self.linear_qkv(hidden_states)

        # Split into Q, K, V
        # Shape: [sq, b, (num_heads + 2*num_kv_heads) * head_dim]
        q_size = self.config.num_attention_heads * self.config.kv_channels
        kv_size = self.config.num_query_groups * self.config.kv_channels

        q = qkv[:, :, :q_size]
        k = qkv[:, :, q_size:q_size + kv_size]
        v = qkv[:, :, q_size + kv_size:]

        if output_gate:
            # Project and normalize gate
            gate, _ = self.g_proj(hidden_states)
            gate = self.g_norm(gate)
            return q, k, v, gate

        return q, k, v

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for LinearAttention.

        Args:
            hidden_states (Tensor): Input tensor of shape [sq, b, h]
            attention_mask (Tensor): Attention mask (must be 2D, not 4D causal)
            key_value_states (Optional[Tensor]): Key/value states for cross attention
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s)
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
                Currently used exclusively for inference with dynamic batching and flashinfer RoPE
            attention_bias (Optional[Tensor]): Attention bias
            packed_seq_params (Optional[PackedSeqParams]): Parameters used for THD format
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs
            inference_params (Optional[BaseInferenceContext]): Deprecated parameter name,
                use inference_context instead

        Returns:
            Tuple[Tensor, Tensor]: Attention output and bias
            - output_tensor: shape [sq, b, h]
            - attention_bias: None (not used in linear attention)
        """
        # Validate attention mask - LinearAttention only supports 2D masks
        if attention_mask is not None and attention_mask.dim() == 4:
            raise ValueError(
                "LinearAttention does not support 4D causal attention masks. "
                "Please use 2D attention masks only."
            )

        # Get sequence length for mode selection
        sq = hidden_states.size(0)

        # Select GLA mode based on sequence length
        # Use fused_recurrent for short sequences (<=64), chunk for longer sequences
        mode = 'fused_recurrent' if sq <= 64 else 'chunk'

        # TODO: Project to QKV
        # qkv, _ = self.linear_qkv(hidden_states)

        # TODO: Split QKV into separate tensors
        # q, k, v = split_qkv(qkv)

        # TODO: Apply QK normalization
        # q = self.q_layernorm(q) if self.q_layernorm else q
        # k = self.k_layernorm(k) if self.k_layernorm else k

        # TODO: Apply rotary position embeddings
        # if rotary_pos_emb is not None:
        #     q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)

        # TODO: Handle GQA (grouped query attention)
        # if num_kv_heads < num_query_heads:
        #     k, v = repeat_kv(k, v, num_query_heads // num_kv_heads)

        # TODO: Apply GLA kernel (chunk or fused_recurrent)
        # gla_fn = self.gla_ops[mode]
        # attn_output = gla_fn(q, k, v, self.slope, ...)

        # TODO: Apply GroupRMSNorm and gating
        # g = self.g_proj(hidden_states)
        # g = self.g_norm(g)
        # output = attn_output * g

        # TODO: Apply output projection
        # output, _ = self.linear_proj(output)

        # Placeholder return - just pass through hidden_states
        return hidden_states, None
