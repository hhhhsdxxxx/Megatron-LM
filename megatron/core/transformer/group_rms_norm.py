# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn

from megatron.core.jit import jit_fuser


class GroupRMSNorm(torch.nn.Module):
    """Group RMS Normalization module.

    Divides the hidden dimension into groups and normalizes each group independently
    using RMS normalization. This is used in linear attention mechanisms.

    Args:
        hidden_size (int): The width of input, i.e. hidden size
        group_norm_size (int): Number of groups to divide hidden_size into
        eps (float): epsilon to use for the norm, default to 1e-6
        sequence_parallel (bool): Set to true if sequence parallelism is being used,
            this marks the weights as needing to be allreduced.

    Example:
        >>> norm = GroupRMSNorm(hidden_size=2048, group_norm_size=4)
        >>> x = torch.randn(2, 128, 2048)  # [batch, seq_len, hidden]
        >>> output = norm(x)  # Same shape as input
    """

    def __init__(
        self,
        hidden_size: int,
        group_norm_size: int,
        eps: float = 1e-6,
        sequence_parallel: bool = False,
    ):
        super().__init__()

        # Validate that hidden_size is divisible by group_norm_size
        assert hidden_size % group_norm_size == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"group_norm_size ({group_norm_size})"
        )

        self.hidden_size = hidden_size
        self.group_norm_size = group_norm_size
        self.group_size = hidden_size // group_norm_size
        self.eps = eps

        # Learnable weight parameter
        self.weight = nn.Parameter(torch.ones(hidden_size))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    @jit_fuser
    def _norm(self, x):
        """Apply group-wise RMS normalization.

        Args:
            x: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized tensor of same shape as input
        """
        # Get original shape and dtype
        orig_shape = x.shape
        orig_dtype = x.dtype

        # Reshape: [..., hidden_size] -> [..., group_norm_size, group_size]
        # This divides the hidden dimension into groups
        reshaped = x.view(*orig_shape[:-1], self.group_norm_size, self.group_size)

        # Compute variance per group (on last dimension) in fp32 for stability
        variance = reshaped.float().pow(2).mean(dim=-1, keepdim=True)

        # Normalize: x / sqrt(variance + eps)
        normalized = reshaped.float() * torch.rsqrt(variance + self.eps)

        # Restore original shape and dtype
        return normalized.view(orig_shape).type_as(x)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized and scaled tensor of same shape as input
        """
        # Validate input shape
        assert x.shape[-1] == self.hidden_size, (
            f"Input tensor's last dimension ({x.shape[-1]}) must match "
            f"hidden_size ({self.hidden_size})"
        )

        output = self._norm(x)
        return output * self.weight
