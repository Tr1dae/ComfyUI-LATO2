"""ComfyUI-LATO2: LATO.2 (factorized vertex + topology flow) mesh generation for ComfyUI.

Upstream: https://github.com/LoHhhha/LATO.2 (MIT). The model code is vendored in
vendor/lato2_vendor/ with pure-PyTorch replacements for spconv, flash-attn, torch-scatter,
o_voxel and open3d, so it runs on a stock ComfyUI Python without CUDA builds.
"""

import os
import sys

# Only vendor/ (which holds nothing but the uniquely named `lato2_vendor` package) goes on
# sys.path, so nothing here can shadow ComfyUI modules such as `nodes`.
_vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor not in sys.path:
    sys.path.append(_vendor)

from .nodes import LATO2Extension, comfy_entrypoint  # noqa: E402

__all__ = ["LATO2Extension", "comfy_entrypoint"]
