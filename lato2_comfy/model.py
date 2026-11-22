"""Loading, device placement and the generation pipeline for LATO.2."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

log = logging.getLogger("ComfyUI-LATO2")

LATO2_REPO = "0x4c48/LATO.2"
TRELLIS_REPO = "microsoft/TRELLIS-image-large"
DINO_REPO = "facebook/dinov2-with-registers-large"
LATO2_FILES = ("vflow.pt", "vvae.pt", "offset_head.pt", "tflow.pt", "tvae.pt", "voxel_encoder.pt")
SS_FLOW_NAME = "ss_flow_img_dit_L_16l8_fp16"
SS_DECODER_NAME = "ss_dec_conv3d_16l8_fp16"

MODEL_FOLDER = "lato2"


def register_folder():
    path = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER] = ([path], {".pt", ".safetensors", ".json"})
    return path


def model_dir() -> str:
    for p in folder_paths.get_folder_paths(MODEL_FOLDER):
        if os.path.isfile(os.path.join(p, "vflow.pt")):
            return p
    return folder_paths.get_folder_paths(MODEL_FOLDER)[0]


# ----------------------------------------------------------------------------- downloads

def ensure_lato2_weights(root: str, allow_download: bool):
    missing = [f for f in LATO2_FILES if not os.path.isfile(os.path.join(root, f))]
    if missing:
        if not allow_download:
            raise FileNotFoundError(f"LATO2: missing {missing} in {root}. Enable auto_download or fetch them from "
                                    f"https://huggingface.co/{LATO2_REPO}")
        from huggingface_hub import hf_hub_download
        for f in missing:
            log.info("LATO2: downloading %s/%s -> %s", LATO2_REPO, f, root)
            hf_hub_download(LATO2_REPO, f, local_dir=root)


def ensure_dino(root: str, allow_download: bool) -> str:
    d = os.path.join(root, "dinov2-with-registers-large")
    if not os.path.isfile(os.path.join(d, "model.safetensors")):
        if not allow_download:
            raise FileNotFoundError(f"LATO2: missing DINOv2 weights in {d}")
        from huggingface_hub import snapshot_download
        log.info("LATO2: downloading %s -> %s", DINO_REPO, d)
        snapshot_download(DINO_REPO, local_dir=d, allow_patterns=["*.json", "model.safetensors"])
    return d


def ensure_trellis_ss(root: str, allow_download: bool) -> str:
    d = os.path.join(root, "trellis")
    for name in (SS_FLOW_NAME, SS_DECODER_NAME):
        for ext in ("json", "safetensors"):
            rel = f"ckpts/{name}.{ext}"
            if not os.path.isfile(os.path.join(d, rel)):
                if not allow_download:
                    raise FileNotFoundError(f"LATO2: missing TRELLIS sparse-structure file {rel} in {d}")
                from huggingface_hub import hf_hub_download
                log.info("LATO2: downloading %s/%s -> %s", TRELLIS_REPO, rel, d)
                hf_hub_download(TRELLIS_REPO, rel, local_dir=d)
    return d


# ----------------------------------------------------------------------------- checkpoint loading

def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # LATO.2 checkpoints store a plain config dict; this should not trigger
        raise RuntimeError(
            f"LATO2: {path} could not be loaded with weights_only=True ({e}). Refusing to unpickle "
            "arbitrary objects; re-download the file from the official repo."
        ) from e


def load_lato2_module(cls, path):
    ckpt = _torch_load(path)
    model = cls(**ckpt["config"]["args"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval().requires_grad_(False), ckpt["config"]


class DinoV2RegEncoder(torch.nn.Module):
    """DINOv2 ViT-L/14 + registers via transformers; returns layer-normed pre-norm tokens like
    LATO.2's torch.hub DinoV2Encoder (backbone(x, is_training=True)['x_prenorm'])."""

    def __init__(self, path: str, img_res: int = 518):
        super().__init__()
        from transformers import Dinov2WithRegistersModel
        self.backbone = Dinov2WithRegistersModel.from_pretrained(path, torch_dtype=torch.float32).eval()
        self.backbone.requires_grad_(False)
        self.img_res = int(img_res)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(images_uint8))
        if x.dim() == 3:
            x = x[None]
        x = x.permute(0, 3, 1, 2).float().div(255.0).to(self.img_mean.device)
        if x.shape[-2:] != (self.img_res, self.img_res):
            x = F.interpolate(x, (self.img_res, self.img_res), mode="bicubic", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        feats = out.hidden_states[-1]
        return F.layer_norm(feats, feats.shape[-1:])


def _module_bytes(m: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in m.parameters()) + \
        sum(b.numel() * b.element_size() for b in m.buffers())


class LATO2Bundle:
    """All LATO.2 networks. Lives on CPU; `on_device()` moves what a stage needs to the GPU."""

    def __init__(self, root: str, auto_download: bool = True):
        from lato2_vendor.models import (
            OffsetHead, TopologySiTFlow, TopologyVAE, VertexSLatFlowModel, VertexVAE, VoxelFieldConditioner,
        )
        self.root = root
        self.auto_download = auto_download
        ensure_lato2_weights(root, auto_download)
        j = lambda f: os.path.join(root, f)  # noqa: E731
        self.vflow, self.vflow_cfg = load_lato2_module(VertexSLatFlowModel, j("vflow.pt"))
        self.vvae, self.vvae_cfg = load_lato2_module(VertexVAE, j("vvae.pt"))
        self.offset_head, _ = load_lato2_module(OffsetHead, j("offset_head.pt"))
        self.tflow, self.tflow_cfg = load_lato2_module(TopologySiTFlow, j("tflow.pt"))
        self.tvae, _ = load_lato2_module(TopologyVAE, j("tvae.pt"))
        self.voxel_encoder, self.venc_cfg = load_lato2_module(VoxelFieldConditioner, j("voxel_encoder.pt"))

        self.res = int(self.vvae_cfg["resolution"])
        self.min_res = int(self.vvae_cfg["min_resolution"])
        self.latent_dim = int(self.vflow_cfg["latent_dim"])
        self.density_max = float(self.vflow_cfg["max_vertex_num"])
        self.img_res = int(self.vflow_cfg["image_resolution"])
        self.dino_version = self.vflow_cfg["dino_version"]
        ta = self.tflow_cfg["args"]
        self.z_dim = int(ta["z_dim"])
        self.max_vertices = int(ta["max_vertices"])
        self.latent_scale = float(self.tflow_cfg["latent_scale"])
        if int(ta["num_discrete"]) != self.res or int(self.venc_cfg["resolution"]) != self.min_res:
            raise ValueError("LATO2: checkpoint resolutions are inconsistent (vvae / tflow / voxel encoder).")
        if self.dino_version != "dinov2_vitl14_reg":
            raise ValueError(f"LATO2: unsupported DINO conditioning {self.dino_version!r}")

        self.dino = DinoV2RegEncoder(ensure_dino(root, auto_download), self.img_res)
        self._ss = None  # TRELLIS sparse-structure flow + decoder, loaded on demand

    # -- optional TRELLIS stage-0 for image -> scaffold
    def sparse_structure_models(self):
        if self._ss is None:
            from lato2_vendor.models.trellis_ss import load_trellis_model
            d = ensure_trellis_ss(self.root, self.auto_download)
            flow = load_trellis_model(SS_FLOW_NAME, d, "cpu").requires_grad_(False)
            dec = load_trellis_model(SS_DECODER_NAME, d, "cpu").requires_grad_(False)
            self._ss = (flow, dec)
        return self._ss

    def on_device(self, modules, device):
        need = sum(_module_bytes(m) for m in modules) + (2 << 30)
        mm.free_memory(need, device)
        for m in modules:
            m.to(device)

    def offload(self, modules):
        for m in modules:
            m.to("cpu")
        mm.soft_empty_cache()


# ----------------------------------------------------------------------------- pipeline

def _progress(total):
    return comfy.utils.ProgressBar(total)


@torch.no_grad()
def encode_images(bundle: LATO2Bundle, images, device):
    return bundle.dino(np.stack(images)).float()


@torch.no_grad()
def sample_sparse_structure(bundle: LATO2Bundle, cond, steps, cfg_strength, cfg_interval, rescale_t, device):
    """TRELLIS stage 0: DINO tokens -> (B, R, R, R) bool occupancy (TRELLIS Z-up frame)."""
    flow, dec = bundle.sparse_structure_models()
    neg = torch.zeros_like(cond)
    r = flow.resolution
    x = torch.randn(cond.shape[0], flow.in_channels, r, r, r, device=device)
    t_seq = np.linspace(1.0, 0.0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    pbar = _progress(steps)
    for i in range(steps):
        mm.throw_exception_if_processing_interrupted()
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        tv = torch.full((x.shape[0],), 1000.0 * t, device=device, dtype=torch.float32)
        v = flow(x, tv, cond)
        if cfg_interval[0] <= t <= cfg_interval[1]:
            v_u = flow(x, tv, neg)
            v = (1 + cfg_strength) * v - cfg_strength * v_u
        x = x - (t - t_prev) * v
        pbar.update(1)
    return dec(x)[:, 0] > 0


@torch.no_grad()
def generate_mesh(bundle: LATO2Bundle, voxels: torch.Tensor, cond: torch.Tensor, device, vert_num: int,
                  vflow_steps: int, cfg_strength: float, rescale_t: float, tflow_steps: int,
                  inference_threshold: float, edge_threshold: float, fill_open_rings: bool, max_ring_size: int,
                  offload: bool = True):
    """One scaffold (N,3 int coords on min_res^3, Y-up) + DINO cond (1, L, C) -> (verts [-0.5,0.5], faces)."""
    from lato2_vendor.modules.sparse import SparseTensor
    from lato2_vendor.utils.inference import build_voxel_fields, decode_vertices, edges_to_faces

    res = bundle.res
    pbar = _progress(vflow_steps + tflow_steps + 2)

    # ---- stage 1: V-Flow on the 64^3 active voxels -> V-VAE vertex decode
    stage1 = [bundle.vflow, bundle.vvae, bundle.offset_head]
    bundle.on_device(stage1, device)
    try:
        density = torch.full((1,), float(vert_num) / bundle.density_max * 1000.0, device=device)
        coords = torch.cat([torch.zeros((voxels.shape[0], 1), dtype=torch.int32), voxels.int()], 1).to(device)
        neg = torch.zeros_like(cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = SparseTensor(coords=coords, feats=torch.randn(coords.shape[0], bundle.latent_dim, device=device))
            t_seq = np.linspace(1.0, 0.0, vflow_steps + 1)
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
            for i in range(vflow_steps):
                mm.throw_exception_if_processing_interrupted()
                t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
                tv = torch.full((1,), 1000.0 * t, device=device, dtype=torch.float32)
                v_c = bundle.vflow(x, tv, cond, density=density)
                v_u = bundle.vflow(x, tv, neg, density=density)
                x = x - (t - t_prev) * ((1 + cfg_strength) * v_c - cfg_strength * v_u)
                pbar.update(1)
            pred_coords, pred_offsets = decode_vertices(bundle.vvae, bundle.offset_head, x, inference_threshold)
        pbar.update(1)
    finally:
        if offload:
            bundle.offload(stage1)

    vert_int = pred_coords[pred_coords[:, 0] == 0, 1:].long()
    vert_off = pred_offsets[pred_coords[:, 0] == 0].float()
    n = int(vert_int.shape[0])
    if n < 3:
        raise RuntimeError(f"LATO2: V-Flow produced only {n} vertices; try another seed or a larger vertex count.")
    if n > bundle.max_vertices:
        raise RuntimeError(f"LATO2: {n} generated vertices exceed T-Flow max_vertices={bundle.max_vertices}; "
                           "lower vertex_count.")

    # ---- stage 2: T-Flow on the generated vertices -> T-VAE edge decode
    stage2 = [bundle.tflow, bundle.tvae, bundle.voxel_encoder]
    bundle.on_device(stage2, device)
    try:
        verts = vert_int[None].to(device)
        mask = torch.ones((1, n), dtype=torch.bool, device=device)
        field = build_voxel_fields([voxels.long()], bundle.min_res, device)
        cond_vox = bundle.voxel_encoder(field)
        z = torch.randn(1, n, bundle.z_dim, device=device)
        t_seq = np.linspace(0.0, 1.0, tflow_steps + 1)
        for i in range(tflow_steps):
            mm.throw_exception_if_processing_interrupted()
            t, t_next = float(t_seq[i]), float(t_seq[i + 1])
            tv = torch.full((1,), t, device=device, dtype=torch.float32)
            z = z + (t_next - t) * bundle.tflow(z, tv, verts=verts, mask=mask, cond=cond_vox, cond_mask=None)
            pbar.update(1)
        z = z.float() / bundle.latent_scale
        with torch.autocast("cuda", dtype=torch.bfloat16):
            edges = bundle.tvae.decode(z, verts=verts, verts_mask=mask, chunk_size=20000, threshold=edge_threshold)[0]
        pbar.update(1)
    finally:
        if offload:
            bundle.offload(stage2)

    faces = edges_to_faces(edges, n, vert_int.numpy(), fill_open_rings=fill_open_rings, max_ring_size=max_ring_size)
    verts_f = vert_int.numpy().astype(np.float64) / res - 0.5 + vert_off.numpy().astype(np.float64) / (res * 2.0)
    return verts_f.astype(np.float32), np.asarray(faces, dtype=np.int64), n
