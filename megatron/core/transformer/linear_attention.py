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
from megatron.core.utils import get_pg_rank, get_pg_size, divide
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

# Import GLA operators
try:
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
    HAVE_GLA = True
except ImportError:
    chunk_simple_gla = None
    fused_recurrent_simple_gla = None
    HAVE_GLA = False

# Import CP-aware GLA operator
try:
    from fla.ops.lightning_attn import chunk_lightning_attn_cp
    HAVE_CP_GLA = True
except ImportError:
    chunk_lightning_attn_cp = None
    HAVE_CP_GLA = False


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

        # Store CP info for context-parallel support
        self.cp_group = self.pg_collection.cp
        self.cp_size = get_pg_size(self.pg_collection.cp)

        # Build layer-dependent slope tensor
        # The slope varies by layer to provide different decay patterns
        # Reference formula uses 1-indexed global_layer_number:
        #   slope = -base_slope * (1 - (global_layer_number - 1) / (num_layers - 1) + 1e-5)
        # global_layer_number accounts for PP offset so first layer globally = 1
        base_slope = self._build_slope_tensor(config.num_attention_heads)
        num_layers = config.num_layers
        global_layer_number = self.layer_number + (self._pp_layer_offset or 0)
        slope = -base_slope * (1 - (global_layer_number - 1) / max(num_layers - 1, 1) + 1e-5)

        # Register only the TP-local slope slice (like reference)
        num_heads_per_partition = self.num_attention_heads_per_partition
        tp_rank = get_pg_rank(self.pg_collection.tp)
        tp_slope = slope[tp_rank * num_heads_per_partition : (tp_rank + 1) * num_heads_per_partition]
        self.register_buffer('slope', tp_slope, persistent=False)

        # MEG-2: Initialize GroupRMSNorm with TP-local size
        # g_proj uses gather_output=False, so its output and attn_output are TP-local.
        # g_norm must match the TP-local dimension (num_heads_per_partition * kv_channels).
        local_gate_size = self.num_attention_heads_per_partition * config.kv_channels
        self.g_norm = GroupRMSNorm(
            hidden_size=local_gate_size,
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

        # Override KV projection sizing with linear_attn_num_query_groups and kv_expand
        linear_attn_num_query_groups = (
            config.linear_attn_num_query_groups
            if config.linear_attn_num_query_groups > 0
            else config.num_query_groups
        )
        world_size = get_pg_size(self.pg_collection.tp)
        self.num_query_groups_per_partition = divide(linear_attn_num_query_groups, world_size)

        # Initialize QKV projection (required by get_query_key_value_tensors)
        # P1 Fix: Use global head counts because ColumnParallelLinear automatically shards output_size
        # Using per-partition counts would cause double-partitioning in TP environments
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.kv_expand * linear_attn_num_query_groups
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        self.linear_qkv = submodules.linear_qkv(
            config.hidden_size,
            self.linear_qkv_out_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear or config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )

        # Initialize QK layernorm if enabled
        if submodules.q_layernorm is not None:
            self.q_layernorm = submodules.q_layernorm(
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = submodules.k_layernorm(
                hidden_size=self.hidden_size_per_attention_head,
                config=config,
                eps=config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        # Internal RotaryEmbedding for LinearAttention
        # In hybrid MLA mode, gpt_model.py skips creating the external RoPE when
        # config.multi_latent_attention is True (MLA creates its own internal RoPE).
        # LinearAttention needs its own internal RoPE to ensure position embeddings
        # are applied regardless of MLA configuration.
        # Reference: LinearAttention always uses non-interleaved RoPE (rotate_half),
        # so we hardcode rotary_interleaved=False.
        self.rotary_interleaved = False
        if getattr(config, 'position_embedding_type', 'rope') == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=getattr(config, 'rotary_percent', 1.0),
                rotary_interleaved=False,
                rotary_base=getattr(config, 'rotary_base', 10000),
                use_cpu_initialization=config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )
        else:
            self.rotary_pos_emb = None

    def backward_dw(self) -> None:
        """Execute weight update operations for all projections.

        MEG-6: LinearAttention inherits from Attention (which has no backward_dw),
        not SelfAttention. We explicitly call backward_dw on each projection to
        ensure g_proj participates in the delayed-wgrad path alongside linear_qkv
        and linear_proj.
        """
        self.linear_qkv.backward_dw()
        self.linear_proj.backward_dw()
        self.g_proj.backward_dw()

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
        # Project to QKV (shared by both split and non-split paths)
        qkv, _ = self.linear_qkv(hidden_states)

        # Apply SiLU activation to QKV projections if configured
        if self.config.use_linear_silu:
            qkv = torch.nn.functional.silu(qkv)

        if not split_qkv:
            # Return packed QKV with split indices
            split_indices = [
                self.num_attention_heads_per_partition * self.config.kv_channels,
                self.num_query_groups_per_partition * self.config.kv_channels * self.config.kv_expand,
                self.num_query_groups_per_partition * self.config.kv_channels * self.config.kv_expand,
            ]
            return qkv, split_indices

        # Split into Q, K, V using TP-local head counts
        # Shape: [sq, b, (num_heads_per_partition + 2*num_kv_heads_per_partition) * head_dim]
        # Use TP-local head counts since QKV projection is tensor-parallel (gather_output=False)
        q_size = self.num_attention_heads_per_partition * self.config.kv_channels
        kv_size = self.num_query_groups_per_partition * self.config.kv_channels * self.config.kv_expand

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
        # P2 Fix: Map deprecated inference_params to inference_context
        if inference_params is not None and inference_context is None:
            inference_context = inference_params

        # MEG-1: Validate attention mask - LinearAttention only supports 2D padding masks
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                f"LinearAttention only supports 2D padding masks of shape [batch, seq_len], "
                f"got {attention_mask.dim()}D mask with shape {attention_mask.shape}."
            )

        # Handle packed sequence params for THD format
        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded if packed_seq_params.cu_seqlens_q_padded is not None else packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded if packed_seq_params.cu_seqlens_kv_padded is not None else packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # Get sequence length for mode selection
        sq = hidden_states.size(0)

        # Select GLA mode: always use chunk during training to avoid shape issues
        # with short sequences; only use fused_recurrent during inference
        if inference_context is not None and sq <= 64:
            mode = 'fused_recurrent'
        else:
            mode = 'chunk'

        # Project to QKV and split
        q, k, v = self.get_query_key_value_tensors(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            output_gate=False,
            split_qkv=True,
        )

        # Get dimensions for reshaping
        sq, b = hidden_states.size(0), hidden_states.size(1)

        # P1 Fix: Reshape Q/K before applying QK layernorm
        # q and k are currently [sq, b, num_heads * head_dim]
        # Reshape to [sq, b, num_heads, head_dim] before normalization
        kv_head_dim = self.hidden_size_per_attention_head * self.config.kv_expand
        q = q.view(sq, b, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        k = k.view(sq, b, self.num_query_groups_per_partition, kv_head_dim)

        # Apply QK normalization if enabled (now with correct shape)
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
        if self.k_layernorm is not None:
            k = self.k_layernorm(k)

        # P1 Fix: Apply RoPE on per-head Q/K tensors
        # Keep q and k in [sq, b, num_heads, head_dim] shape for RoPE
        # Megatron's RoPE helpers expect per-head layout where last dim is head_dim
        #
        # LinearAttention always uses non-interleaved RoPE (rotate_half, Llama-style),
        # matching the reference apply_rotary_pos_emb() which uses rotate_half internally.
        # When no external RoPE is provided (hybrid MLA mode), compute internally.
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            # Use cos/sin directly
            from megatron.core.models.common.embeddings.rope_utils import (
                apply_rotary_pos_emb_with_cos_sin,
            )

            q = apply_rotary_pos_emb_with_cos_sin(
                q, rotary_pos_cos, rotary_pos_sin, rotary_interleaved=self.rotary_interleaved
            )
            k = apply_rotary_pos_emb_with_cos_sin(
                k, rotary_pos_cos, rotary_pos_sin, rotary_interleaved=self.rotary_interleaved
            )
        elif rotary_pos_emb is not None:
            # Use rotary_pos_emb (freqs)
            # Use lower-level _apply_rotary_pos_emb_bshd to override rotary_interleaved
            from megatron.core.models.common.embeddings.rope_utils import (
                _apply_rotary_pos_emb_bshd,
            )

            # Handle tuple format (q_pos_emb, k_pos_emb)
            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                q = _apply_rotary_pos_emb_bshd(
                    q,
                    q_pos_emb,
                    rotary_interleaved=self.rotary_interleaved,
                )
            if k_pos_emb is not None:
                k = _apply_rotary_pos_emb_bshd(
                    k,
                    k_pos_emb,
                    rotary_interleaved=self.rotary_interleaved,
                )
        elif self.rotary_pos_emb is not None:
            # No external RoPE provided (hybrid MLA mode) — compute internally
            # Use forward() which handles CP-aware position remapping
            from megatron.core.models.common.embeddings.rope_utils import (
                _apply_rotary_pos_emb_bshd,
            )

            rotary_seq_len = sq
            if inference_context is not None:
                if hasattr(inference_context, 'sequence_len_offset'):
                    rotary_seq_len += inference_context.sequence_len_offset
                elif sequence_len_offset is not None:
                    rotary_seq_len += sequence_len_offset
            # forward() returns CP-aware freqs of shape [seq, 1, 1, dim]
            freqs = self.rotary_pos_emb(rotary_seq_len)
            q = _apply_rotary_pos_emb_bshd(
                q, freqs, rotary_interleaved=self.rotary_interleaved,
            )
            k = _apply_rotary_pos_emb_bshd(
                k, freqs, rotary_interleaved=self.rotary_interleaved,
            )

        # Flatten q and k back to [sq, b, num_heads * head_dim] after RoPE
        q = q.view(sq, b, -1)
        k = k.view(sq, b, -1)

        # MEG-2: Reshape Q, K, V for GLA kernel using TP-local head counts
        # QKV tensors are already TP-sharded (gather_output=False), so we must use
        # partition-local head counts end-to-end to avoid shape mismatches under TP > 1.
        # Megatron format: [sq, b, num_heads_local * head_dim]
        # GLA format: [b, sq, num_heads_local, head_dim]
        sq, b = q.size(0), q.size(1)
        num_heads = self.num_attention_heads_per_partition
        num_kv_heads = self.num_query_groups_per_partition
        head_dim = self.hidden_size_per_attention_head
        kv_head_dim = head_dim * self.config.kv_expand

        # Reshape Q: [sq, b, num_heads_local * head_dim] -> [b, sq, num_heads_local, head_dim]
        q = q.view(sq, b, num_heads, head_dim).transpose(0, 1).contiguous()

        # Reshape K, V: [sq, b, num_kv_heads_local * kv_head_dim] -> [b, sq, num_kv_heads_local, kv_head_dim]
        k = k.view(sq, b, num_kv_heads, kv_head_dim).transpose(0, 1).contiguous()
        v = v.view(sq, b, num_kv_heads, kv_head_dim).transpose(0, 1).contiguous()

        # Handle GQA (grouped query attention) - expand K, V if needed
        if num_kv_heads < num_heads:
            # Repeat K, V to match num_heads
            # [b, sq, num_kv_heads, head_dim] -> [b, sq, num_heads, head_dim]
            num_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        # Handle inference context (KV cache for recurrent state)
        recurrent_state = None
        output_final_state = False

        if inference_context is not None:
            # For LinearAttention, we store recurrent_state instead of K/V tensors
            # The recurrent_state has shape [batch, num_heads, head_dim, head_dim]
            output_final_state = True

            # P1 Fix: Reset recurrent state when new inference sequence starts
            # Check if this is the first token of a new sequence (sequence_len_offset == 0)
            # to avoid contaminating outputs with previous request's state
            is_first_token = False
            if hasattr(inference_context, 'sequence_len_offset'):
                is_first_token = inference_context.sequence_len_offset == 0
            elif sequence_len_offset is not None:
                is_first_token = sequence_len_offset == 0

            # Retrieve cached recurrent state from inference context
            if hasattr(inference_context, 'key_value_memory_dict') and not is_first_token:
                layer_key = f'layer_{self.layer_number}'
                if layer_key in inference_context.key_value_memory_dict:
                    cached_state = inference_context.key_value_memory_dict[layer_key]
                    if cached_state is not None:
                        # P2 Fix: Extract recurrent_state from tuple format for compatibility
                        # StaticInferenceContext utilities expect KV-style tuples
                        recurrent_state = cached_state[0] if isinstance(cached_state, tuple) else cached_state
                        # Ensure recurrent_state is on the same device
                        if recurrent_state.device != hidden_states.device:
                            recurrent_state = recurrent_state.to(hidden_states.device).contiguous()

        # Handle left-padding for first generation step
        # When recurrent_state is None (first step), apply attention_mask to value_states
        # to zero out padded positions
        if recurrent_state is None and attention_mask is not None and output_final_state:
            # attention_mask shape: [batch, seq_len] with 0 for padding, 1 for valid
            # Expand to match v shape: [b, sq, num_heads, head_dim]
            # Use in-place multiplication for efficiency
            mask_expanded = attention_mask[:, -sq:, None, None]  # [b, sq, 1, 1]
            v = v * mask_expanded

        # Apply GLA kernel
        gla_fn = self.gla_ops[mode]

        # self.slope is already TP-local (sliced in __init__)
        local_slope = self.slope
        slope_expanded = local_slope[None, None, :].expand(b, sq, num_heads)

        # Call GLA kernel with CP branching
        cp_size = get_pg_size(self.pg_collection.cp)
        if cp_size <= 1:
            attn_output, recurrent_state = gla_fn(
                q=q,
                k=k,
                v=v,
                g=slope_expanded,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens_q,
            )
        else:
            if not HAVE_CP_GLA:
                raise ImportError(
                    "CP-aware GLA operator not found. Please install fla with "
                    "chunk_lightning_attn_cp support."
                )
            attn_output, recurrent_state = chunk_lightning_attn_cp(
                q=q,
                k=k,
                v=v,
                g_gamma=local_slope,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cp_group=self.cp_group,
            )

        # Store recurrent state back to inference context if needed
        if output_final_state and inference_context is not None:
            if hasattr(inference_context, 'key_value_memory_dict'):
                layer_key = f'layer_{self.layer_number}'
                # P2 Fix: Store recurrent_state as (recurrent_state, None) tuple
                # This maintains compatibility with StaticInferenceContext utilities
                # that expect KV-style tuples for swap_key_value_dict() and __eq__()
                inference_context.key_value_memory_dict[layer_key] = (recurrent_state, None)

        # Reshape output back to Megatron format
        # GLA output: [b, sq, num_heads, head_dim]
        # Megatron format: [sq, b, num_heads * head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(sq, b, num_heads * head_dim)

        # Apply GroupRMSNorm and gating
        # IMPORTANT: Apply g_norm to attention output (not gate), matching reference implementation
        # Reference: o = g_norm(o); g_proj = g_proj(hidden); o = o * sigmoid(g_proj)
        attn_output = self.g_norm(attn_output)

        # Project hidden_states to get gate values
        gate, _ = self.g_proj(hidden_states)

        # Apply gating activation — always sigmoid (reference alignment)
        attn_output = attn_output * torch.sigmoid(gate)

        # Apply output projection
        output, output_bias = self.linear_proj(attn_output)

        # Return output and bias (matching Attention interface)
        return output, output_bias
