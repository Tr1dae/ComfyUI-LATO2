# ComfyUI-LATO2: pure-PyTorch replacement for the spconv SubMConv3d used by LATO.2.
#
# LATO.2 only uses stride-1 submanifold convolutions, so a gather + GEMM over a
# precomputed neighbour map reproduces spconv exactly (same weight layout:
# (Co, K0, K1, K2, Ci), cross-correlation over coords[:, 1:4]). No CUDA build needed.

from typing import Optional

import torch
import torch.nn as nn

from .. import SparseTensor

__all__ = ["SparseConv3d", "SparseInverseConv3d"]

# Bound the (chunk, K^3 * Ci) gather buffer.
_GATHER_BYTES = 512 * 1024 * 1024
# Bound the (chunk, K^3) int64 candidate buffer when building neighbour maps.
_NEIGHBOR_ROWS = 1 << 18


def _kernel_offsets(kernel_size, dilation, device):
    k0, k1, k2 = kernel_size
    d0, d1, d2 = dilation
    r0 = (torch.arange(k0, device=device) - k0 // 2) * d0
    r1 = (torch.arange(k1, device=device) - k1 // 2) * d1
    r2 = (torch.arange(k2, device=device) - k2 // 2) * d2
    g = torch.stack(torch.meshgrid(r0, r1, r2, indexing="ij"), dim=-1)
    return g.reshape(-1, 3).long()  # (V, 3), same order as weight.view(Co, V, Ci)


@torch.no_grad()
def build_neighbor_map(coords: torch.Tensor, kernel_size, dilation) -> torch.Tensor:
    """neighbor[i, v] = row of the voxel at coords[i] + offset[v], or -1."""
    device = coords.device
    c = coords.long()
    n = c.shape[0]
    offs = _kernel_offsets(kernel_size, dilation, device)
    pad = offs.abs().max().item() if offs.numel() else 0
    lo = c[:, 1:].min(0).values - pad
    ext = c[:, 1:].max(0).values - lo + pad + 1
    e0, e1, e2 = [int(x) for x in ext.tolist()]

    def key(b, x, y, z):
        return ((b * e0 + x) * e1 + y) * e2 + z

    base = c[:, 1:] - lo
    keys = key(c[:, 0], base[:, 0], base[:, 1], base[:, 2])
    sorted_keys, order = torch.sort(keys)

    V = offs.shape[0]
    out = torch.empty((n, V), dtype=torch.int64, device=device)
    for s in range(0, n, _NEIGHBOR_ROWS):
        e = min(s + _NEIGHBOR_ROWS, n)
        cand = base[s:e, None, :] + offs[None]  # (c, V, 3); stays >= 0 thanks to pad
        q = key(c[s:e, 0:1], cand[..., 0], cand[..., 1], cand[..., 2]).reshape(-1)
        idx = torch.searchsorted(sorted_keys, q).clamp_(max=n - 1)
        hit = sorted_keys[idx] == q
        res = torch.where(hit, order[idx], torch.full_like(idx, -1))
        out[s:e] = res.view(e - s, V)
    return out


def submanifold_conv3d(feats, neighbor, weight, bias):
    n = feats.shape[0]
    co = weight.shape[0]
    ci = weight.shape[-1]
    V = neighbor.shape[1]
    w = weight.reshape(co, V * ci).t()  # (V*Ci, Co)
    out = torch.empty((n, co), dtype=feats.dtype, device=feats.device)
    if n == 0:
        return out
    if V == 1:
        # spconv short-circuits 1x1x1 kernels to a matmul that *reinterprets* the
        # [Co, 1, 1, 1, Ci] buffer as [Ci, Co] (no transpose) - unlike its KRSC layout
        # for k > 1. The released LATO.2 checkpoints were trained through that quirk
        # (V-VAE decoder skip connections), so reproduce it. Same fix as upstream's
        # own torch backend in the LATO.2 Hugging Face Space.
        out = feats @ weight.reshape(ci, co)
    else:
        chunk = max(1, min(n, _GATHER_BYTES // max(1, V * ci * feats.element_size())))
        zero_row = torch.zeros((1, ci), dtype=feats.dtype, device=feats.device)
        padded = torch.cat([feats, zero_row], dim=0)  # index n -> zeros
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            nb = neighbor[s:e]
            nb = torch.where(nb < 0, torch.full_like(nb, n), nb)
            g = padded[nb].reshape(e - s, V * ci)
            out[s:e] = g @ w
    if bias is not None:
        out = out + bias
    return out


class _SubMConv3dParams(nn.Module):
    """Holds weights under the same names / shapes as spconv.SubMConv3d."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)


def _triple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x, x, x)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        if any(s != 1 for s in _triple(stride)) or padding is not None:
            raise NotImplementedError(
                "ComfyUI-LATO2 torch sparse backend only implements stride-1 submanifold conv "
                "(all LATO.2 checkpoints use only that)."
            )
        self.conv = _SubMConv3dParams(in_channels, out_channels, _triple(kernel_size), _triple(dilation), bias)
        self.stride = (1, 1, 1)
        self.padding = None

    def forward(self, x: SparseTensor) -> SparseTensor:
        conv = self.conv
        V = conv.kernel_size[0] * conv.kernel_size[1] * conv.kernel_size[2]
        cache_key = f"torch_subm_{conv.kernel_size}_{conv.dilation}"
        neighbor = None
        if V > 1:
            cached = x.get_spatial_cache(cache_key)
            # Only reuse a map that was built for this exact coordinate tensor.
            if cached is not None and cached[0] is x.coords:
                neighbor = cached[1]
            else:
                neighbor = build_neighbor_map(x.coords, conv.kernel_size, conv.dilation)
                x.register_spatial_cache(cache_key, (x.coords, neighbor))
        else:
            neighbor = torch.empty((x.feats.shape[0], 1), dtype=torch.int64, device=x.device)

        feats = x.feats
        dtype = feats.dtype
        # spconv path ran in fp32; keep that for numerical parity.
        weight = conv.weight.to(device=feats.device, dtype=torch.float32)
        bias = conv.bias.to(device=feats.device, dtype=torch.float32) if conv.bias is not None else None
        with torch.autocast(device_type=feats.device.type, enabled=False):
            out = submanifold_conv3d(feats.float().reshape(feats.shape[0], -1), neighbor, weight, bias)
        return x.replace(out.to(dtype))


class SparseInverseConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("SparseInverseConv3d is not used by LATO.2 inference.")
