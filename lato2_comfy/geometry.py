"""Mesh / voxel / image preprocessing for LATO.2, without open3d, o_voxel or nvdiffrast.

Frames: LATO.2 works Y-up in the unit cube [-0.5, 0.5]^3 with vertices quantised on a
1024^3 grid and a 64^3 active-voxel scaffold (the same frame as ComfyUI MESH outputs
of the native Trellis2 decoders after their Z-up -> Y-up rotation).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ----------------------------------------------------------------------------- mesh quantisation

def normalize_to_unit_cube(vertices: np.ndarray):
    """bbox-centre and scale so the longest side is 1. Returns (v01 in [0,1], center, extent)."""
    vmin, vmax = vertices.min(axis=0), vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = max(float((vmax - vmin).max()), 1e-7)
    return (vertices - center) / extent + 0.5, center, extent


def quantize_mesh_clustering(vertices: np.ndarray, faces: np.ndarray, resolution: int = 1024):
    """Array version of lato2 dataset.utils.quantize_mesh_clustering.

    Returns (v_int [N,3] int32, faces [M,3] int64 on the clustered vertices, center, extent).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normalized_v, center, extent = normalize_to_unit_cube(vertices)

    v_grid = np.clip(np.floor(normalized_v * resolution), 0, resolution - 1).astype(np.int64)
    v_hash = (v_grid[:, 0] * resolution + v_grid[:, 1]) * resolution + v_grid[:, 2]
    unique_hashes, inverse = np.unique(v_hash, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = len(unique_hashes)

    v_sum = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(v_sum, inverse, normalized_v)
    np.add.at(counts, inverse, 1)
    v_mean = v_sum / counts[:, None]
    v_int = np.clip(np.floor(v_mean * resolution), 0, resolution - 1).astype(np.int32)

    new_faces = inverse[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    return v_int, new_faces[valid], center, extent


# ----------------------------------------------------------------------------- voxelisation

def _tri_box_overlap(tri: torch.Tensor, centers: torch.Tensor, half: float) -> torch.Tensor:
    """Separating-axis triangle / axis-aligned cube test (Akenine-Moller).

    tri: (N, 3, 3) triangle vertices, centers: (N, 3) box centres, half: box half-size.
    """
    v = tri - centers[:, None, :]
    v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]
    e = [v1 - v0, v2 - v1, v0 - v2]
    ok = torch.ones(tri.shape[0], dtype=torch.bool, device=tri.device)

    # 1) box face normals (AABB of the triangle vs. the box)
    vmin = torch.minimum(torch.minimum(v0, v1), v2)
    vmax = torch.maximum(torch.maximum(v0, v1), v2)
    ok &= (vmin <= half).all(-1) & (vmax >= -half).all(-1)

    # 2) triangle normal
    n = torch.cross(e[0], e[1], dim=-1)
    d = (n * v0).sum(-1)
    r = half * n.abs().sum(-1)
    ok &= d.abs() <= r

    # 3) nine edge x axis cross products
    axes = torch.eye(3, device=tri.device, dtype=tri.dtype)
    for ed in e:
        for a in range(3):
            ax = torch.cross(ed, axes[a].expand_as(ed), dim=-1)
            p0, p1, p2 = (ax * v0).sum(-1), (ax * v1).sum(-1), (ax * v2).sum(-1)
            pmin = torch.minimum(torch.minimum(p0, p1), p2)
            pmax = torch.maximum(torch.maximum(p0, p1), p2)
            rad = half * ax.abs().sum(-1)
            ok &= ~((pmin > rad) | (pmax < -rad))
    return ok


def _subdivide_large(tri: torch.Tensor, max_extent: float) -> torch.Tensor:
    """Split triangles (4-way midpoint) until each bbox edge is <= max_extent."""
    out = []
    for _ in range(16):
        ext = (tri.max(1).values - tri.min(1).values).max(-1).values
        big = ext > max_extent
        out.append(tri[~big])
        if not bool(big.any()):
            break
        t = tri[big]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = torch.cat([
            torch.stack([a, ab, ca], 1), torch.stack([ab, b, bc], 1),
            torch.stack([ca, bc, c], 1), torch.stack([ab, bc, ca], 1),
        ])
    else:
        out.append(tri)
    return torch.cat(out)


@torch.no_grad()
def voxelize_surface(vertices: np.ndarray, faces: np.ndarray, resolution: int = 64,
                     device: torch.device | str = "cpu", chunk: int = 1 << 19) -> torch.Tensor:
    """Voxels of a resolution^3 grid over [-0.5, 0.5]^3 that the surface passes through.

    Replaces o_voxel.convert.mesh_to_flexible_dual_grid(...)[0] as used by LATO.2.
    Returns sorted (N, 3) int32 coords.
    """
    v = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=device) * 0.99999
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=device)
    if f.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.int32)
    # grid units: voxel i spans [i, i+1)
    tri = (v[f] + 0.5) * resolution
    tri = _subdivide_large(tri, 1.0)

    offs = torch.stack(torch.meshgrid(*([torch.arange(2, device=device)] * 3), indexing="ij"), -1).reshape(-1, 3)
    found = []
    for s in range(0, tri.shape[0], chunk):
        t = tri[s:s + chunk]
        base = torch.floor(t.min(1).values).long()  # bbox <= 1 voxel wide -> 2x2x2 candidates
        cand = base[:, None, :] + offs[None]  # (n, 8, 3)
        tt = t[:, None].expand(-1, 8, -1, -1).reshape(-1, 3, 3)
        cc = cand.reshape(-1, 3)
        inside = ((cc >= 0) & (cc < resolution)).all(-1)
        hit = inside & _tri_box_overlap(tt, cc.float() + 0.5, 0.5 + 1e-6)
        hc = cc[hit]
        found.append(torch.unique((hc[:, 0] * resolution + hc[:, 1]) * resolution + hc[:, 2]))
    keys = torch.unique(torch.cat(found))
    coords = torch.stack([keys // (resolution * resolution), (keys // resolution) % resolution, keys % resolution], -1)
    return coords.int().cpu()


def dense_to_coords(occ: torch.Tensor) -> torch.Tensor:
    """(R, R, R) bool -> sorted (N, 3) int32."""
    return torch.argwhere(occ).int()


def zup_to_yup_coords(coords: torch.Tensor, res: int) -> torch.Tensor:
    """TRELLIS / Trellis2 voxel frame (Z-up) -> LATO.2 (Y-up): (x, y, z) -> (x, z, R-1-y)."""
    c = torch.stack([coords[:, 0], coords[:, 2], res - 1 - coords[:, 1]], dim=1)
    order = torch.argsort((c[:, 0].long() * res + c[:, 1]) * res + c[:, 2])
    return c[order].int()


# ----------------------------------------------------------------------------- conditioning images

def crop_foreground(rgb: np.ndarray, alpha: np.ndarray, padding: float = 1.2, size: int = 518) -> np.ndarray:
    """Square crop around alpha > 0.8 and composite on black (LATO.2 utils.image.crop_foreground).

    rgb: (H, W, 3) float [0,1], alpha: (H, W) float [0,1]. Returns uint8 (size, size, 3).
    """
    ys, xs = np.nonzero(alpha > 0.8)
    if xs.size == 0:
        raise ValueError("LATO2: no foreground found in the mask (mask > 0.8).")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(max(x1 - x0, y1 - y0) * padding)
    if side < 2:
        raise ValueError("LATO2: foreground is too small to crop.")
    rgba = np.concatenate([rgb, alpha[..., None]], -1)
    pil = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    box = (int(cx - side // 2), int(cy - side // 2), int(cx + side // 2), int(cy + side // 2))
    out = pil.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    out = np.asarray(out).astype(np.float32) / 255
    out = out[:, :, :3] * out[:, :, 3:4]
    return (out * 255).astype(np.uint8)


def _orbit_camera(center, distance, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    h = distance * math.cos(el)
    # LATO2 WhiteModelRenderer with up_axis="y": plane axes are x and z
    eye = center + torch.tensor([h * math.cos(az), distance * math.sin(el), h * math.sin(az)],
                                device=center.device, dtype=center.dtype)
    return eye


@torch.no_grad()
def render_white_model(vertices: np.ndarray, faces: np.ndarray, azimuth: float = 45.0, elevation: float = 30.0,
                       img_res: int = 518, render_res: int = 1024, device: torch.device | str = "cuda",
                       crop_padding: float = 1.2) -> np.ndarray:
    """Clay render matching LATO.2's WhiteModelRenderer framing (fov 50, distance 1.8, Y-up orbit,
    black background, cropped to the object with 1.2 padding). Ray-cast with ComfyUI's BVH.

    Lighting is an approximation of the upstream Filament sun + IBL setup. Returns uint8 (img_res, img_res, 3).
    """
    from comfy_extras.nodes_mesh_postprocess import (
        _barycentric, _build_triangle_bvh, _camera_basis, _closest_hit_rays_bvh, _view_rays,
    )

    v = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=device)
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=device)
    tri = v[f]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)  # area-weighted
    vn = torch.zeros_like(v).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
    vn = F.normalize(vn, dim=-1, eps=1e-8)

    center = (v.min(0).values + v.max(0).values) / 2
    eye = _orbit_camera(center, 1.8, azimuth, elevation)
    fwd, right, up = _camera_basis(eye, center, torch.tensor([0.0, 1.0, 0.0], device=device))
    H = W = int(render_res)
    o, d = _view_rays(eye, fwd, right, up, math.radians(50.0), H, W)

    bvh = _build_triangle_bvh(tri)
    img = torch.zeros((H * W, 3), device=device)
    mask = torch.zeros(H * W, dtype=torch.bool, device=device)
    albedo = torch.tensor([0.78, 0.78, 0.82], device=device)
    sun = F.normalize(torch.tensor([-0.35, 1.0, -0.35], device=device), dim=0)  # towards the light
    step = 1 << 20
    for s in range(0, H * W, step):
        e = min(s + step, H * W)
        t_hit, face, hit = _closest_hit_rays_bvh(o[s:e], d[s:e], tri, bvh, tmin=1e-5, tmax=1e30)
        if not bool(hit.any()):
            continue
        fh = face[hit].clamp_min(0)
        P = o[s:e][hit] + t_hit[hit, None] * d[s:e][hit]
        bary = _barycentric(P, tri[fh])
        n = F.normalize((bary[:, :, None] * vn[f[fh]]).sum(1), dim=-1, eps=1e-8)
        n = torch.where((n * d[s:e][hit]).sum(-1, keepdim=True) > 0, -n, n)  # face the camera
        diffuse = (n * sun).sum(-1).clamp_min(0)
        sky = 0.5 + 0.5 * n[:, 1]  # hemispheric ambient (IBL stand-in)
        lin = albedo * (0.16 + 0.34 * sky + 0.62 * diffuse)[:, None]
        local = torch.zeros((e - s, 3), device=device)
        local[hit] = lin.clamp(0, 1).pow(1 / 2.2)
        img[s:e] = local
        mlocal = torch.zeros(e - s, dtype=torch.bool, device=device)
        mlocal[hit] = True
        mask[s:e] = mlocal

    rgb = img.reshape(H, W, 3).cpu().numpy()
    alpha = mask.reshape(H, W).float().cpu().numpy()
    if alpha.sum() == 0:
        raise ValueError("LATO2: conditioning render is empty (mesh not visible).")
    return crop_foreground(rgb * alpha[..., None], alpha, crop_padding, img_res)


# ----------------------------------------------------------------------------- ComfyUI tensors

def comfy_image_to_np(image: torch.Tensor, index: int = 0) -> np.ndarray:
    """ComfyUI IMAGE (B, H, W, C) float -> (H, W, 3) float32."""
    return image[min(index, image.shape[0] - 1), :, :, :3].float().cpu().numpy()


def np_uint8_to_comfy_image(arrs) -> torch.Tensor:
    return torch.from_numpy(np.stack([a.astype(np.float32) / 255.0 for a in arrs]))
