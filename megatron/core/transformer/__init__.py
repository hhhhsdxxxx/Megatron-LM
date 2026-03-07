# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from .module import MegatronModule
from .spec_utils import ModuleSpec, build_module
from .transformer_config import MLATransformerConfig, TransformerConfig
from .transformer_layer import TransformerLayer, TransformerLayerSubmodules

# Import LinearAttention after TransformerConfig to avoid circular dependency
from .linear_attention import LinearAttention
