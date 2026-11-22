"""ComfyUI nodes for LATO.2 (factorized vertex + topology flow mesh generation)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, IO, Types, io

from .lato2_comfy import geometry as geo
from .lato2_comfy.model import (
    LATO2Bundle, encode_images, generate_mesh, model_dir, register_folder, sample_sparse_structure,
)

log = logging.getLogger("ComfyUI-LATO2")

LATO2Model = io.Custom("LATO2_MODEL")
LATO2Scaffold = io.Custom("LATO2_SCAFFOLD")

_CACHE: dict = {}


class Scaffold:
    """A 64^3 active-voxel scaffold in LATO.2's Y-up unit-cube frame, plus how to map results back."""

    def __init__(self, voxels: torch.Tensor, resolution: int, cond_image: np.ndarray | None,
                 center=None, extent: float = 1.0, source: str = ""):
        self.voxels = voxels.int().cpu()  # (N, 3)
        self.resolution = int(resolution)
        self.cond_image = cond_image  # uint8 (518, 518, 3) or None
        self.center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
        self.extent = float(extent)
        self.source = source

    def _comfy_cache_tensors(self):
        return (self.voxels,)


def _first_mesh(mesh):
    from comfy_extras.nodes_save_3d import get_mesh_batch_item
    if int(mesh.vertices.shape[0]) > 1:
        log.warning("LATO2: mesh batch of %d, using the first item.", int(mesh.vertices.shape[0]))
    v, f, *_ = get_mesh_batch_item(mesh, 0)
    return v.float().cpu().numpy(), f.long().cpu().numpy()


def _cond_from_image(image, mask, crop_padding, size):
    rgb = geo.comfy_image_to_np(image)
    if mask is not None:
        m = mask[0] if mask.dim() == 3 else mask
        m = m.float().cpu()
        if m.shape != rgb.shape[:2]:
            m = torch.nn.functional.interpolate(m[None, None], size=rgb.shape[:2], mode="bilinear")[0, 0]
        return geo.crop_foreground(rgb, m.numpy(), crop_padding, size)
    # no mask: assume a pre-cropped subject (e.g. ImageCropToMask with a black background)
    from PIL import Image
    pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(pil)


# ============================================================================= loader

class LATO2LoadModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2LoadModel",
            display_name="LATO2 Load Model",
            category="3d/LATO2",
            description="Loads the LATO.2 V-Flow / V-VAE / T-Flow / T-VAE networks and the DINOv2 encoder "
                        "from models/lato2 (downloaded from Hugging Face on first use).",
            inputs=[
                IO.Boolean.Input("auto_download", default=True,
                                 tooltip="Fetch missing weights from huggingface.co/0x4c48/LATO.2 (~3.4 GB), "
                                         "facebook/dinov2-with-registers-large (~1.2 GB) and, for image input, "
                                         "the TRELLIS sparse-structure stage."),
                IO.Boolean.Input("offload_after_run", default=True,
                                 tooltip="Move LATO.2 back to system RAM after each stage so Trellis2 & co. keep their VRAM."),
            ],
            outputs=[LATO2Model.Output(display_name="lato2_model")],
        )

    @classmethod
    def execute(cls, auto_download, offload_after_run):
        root = model_dir()
        key = root
        bundle = _CACHE.get(key)
        if bundle is None:
            bundle = LATO2Bundle(root, auto_download=auto_download)
            _CACHE.clear()
            _CACHE[key] = bundle
        return IO.NodeOutput({"bundle": bundle, "offload": offload_after_run})


# ============================================================================= scaffolds

class LATO2MeshToScaffold(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2MeshToScaffold",
            display_name="LATO2 Mesh To Scaffold",
            category="3d/LATO2",
            description="Voxelizes any mesh (Trellis2, Pixal3D, Hunyuan3D, a loaded GLB...) into LATO.2's 64^3 "
                        "scaffold and renders the clay conditioning view LATO.2 was trained with.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("render_azimuth", default=45.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Orbit azimuth of the conditioning render (upstream default 45)."),
                IO.Float.Input("render_elevation", default=30.0, min=-89.0, max=89.0, step=1.0,
                               tooltip="Orbit elevation of the conditioning render (upstream default 30)."),
            ],
            outputs=[
                LATO2Scaffold.Output(display_name="scaffold"),
                IO.Image.Output(display_name="cond_render"),
            ],
        )

    @classmethod
    def execute(cls, mesh, render_azimuth, render_elevation):
        device = mm.get_torch_device()
        v, f = _first_mesh(mesh)
        if len(f) == 0:
            raise ValueError("LATO2: input mesh has no faces.")
        res = 1024
        v_int, faces_q, center, extent = geo.quantize_mesh_clustering(v, f, res)
        quant_v = np.clip(v_int.astype(np.float64) / (res - 1.0) - 0.5, -0.5 + 1e-6, 0.5 - 1e-6)
        voxels = geo.voxelize_surface(quant_v, faces_q, 64, device=device)
        render_v = v_int.astype(np.float64) / res - 0.5
        img = geo.render_white_model(render_v, faces_q, render_azimuth, render_elevation, 518, device=device)
        log.info("LATO2: mesh %d verts / %d faces -> %d active voxels", len(v), len(f), voxels.shape[0])
        sc = Scaffold(voxels, 64, img, center=center, extent=extent, source="mesh")
        return IO.NodeOutput(sc, geo.np_uint8_to_comfy_image([img]))


class LATO2VoxelToScaffold(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2VoxelToScaffold",
            display_name="LATO2 Voxel To Scaffold",
            category="3d/LATO2",
            description="Uses a sparse-structure VOXEL (e.g. 'Trellis2 VAE Decode Structure' at resolution 64) as "
                        "the LATO.2 scaffold, skipping the Trellis2 shape stages. Needs a conditioning image on the "
                        "LATO2 Generate node.",
            inputs=[
                IO.Voxel.Input("voxel"),
                IO.Combo.Input("source_frame", options=["z_up (Trellis / Trellis2)", "y_up (Pixal3D / LATO2)"],
                               default="z_up (Trellis / Trellis2)"),
                IO.Int.Input("batch_index", default=0, min=0, max=63),
            ],
            outputs=[LATO2Scaffold.Output(display_name="scaffold")],
        )

    @classmethod
    def execute(cls, voxel, source_frame, batch_index):
        data = voxel.data
        if data.dim() == 5:
            data = data[:, 0]
        if data.dim() == 4:
            occ = data[min(batch_index, data.shape[0] - 1)] > 0.5
        elif data.dim() == 3:
            occ = data > 0.5
        else:
            raise ValueError(f"LATO2: expected a dense (B, R, R, R) structure voxel, got shape {tuple(data.shape)}")
        r = occ.shape[-1]
        if r != 64:
            if 64 % r == 0:
                k = 64 // r
                occ = occ.repeat_interleave(k, 0).repeat_interleave(k, 1).repeat_interleave(k, 2)
                log.warning("LATO2: structure voxel is %d^3; nearest-upsampled to 64^3 (decode at 64 for best results).", r)
            else:
                occ = torch.nn.functional.max_pool3d(occ[None, None].float(), r // 64)[0, 0] > 0.5
        coords = geo.dense_to_coords(occ.cpu())
        if source_frame.startswith("z_up"):
            coords = geo.zup_to_yup_coords(coords, 64)
        if coords.shape[0] == 0:
            raise ValueError("LATO2: the structure voxel is empty.")
        return IO.NodeOutput(Scaffold(coords, 64, None, source="voxel"))


class LATO2ImageToScaffold(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2ImageToScaffold",
            display_name="LATO2 Image To Scaffold (TRELLIS SS)",
            category="3d/LATO2",
            description="Upstream LATO.2 image-to-mesh stage 0: generates the 64^3 scaffold from one image with the "
                        "TRELLIS (v1) sparse-structure flow. Connect a foreground mask (1 = object) or a pre-cropped "
                        "subject on black.",
            inputs=[
                LATO2Model.Input("lato2_model"),
                IO.Image.Input("image"),
                IO.Mask.Input("mask", optional=True, tooltip="Foreground = 1 (e.g. from a background-removal node)."),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                IO.Int.Input("steps", default=25, min=1, max=200),
                IO.Float.Input("cfg", default=5.0, min=0.0, max=20.0, step=0.1),
                IO.Float.Input("crop_padding", default=1.2, min=1.0, max=2.0, step=0.01),
            ],
            outputs=[
                LATO2Scaffold.Output(display_name="scaffold"),
                IO.Image.Output(display_name="cond_image"),
            ],
        )

    @classmethod
    def execute(cls, lato2_model, image, seed, steps, cfg, crop_padding, mask=None):
        bundle: LATO2Bundle = lato2_model["bundle"]
        device = mm.get_torch_device()
        img = _cond_from_image(image, mask, crop_padding, bundle.img_res)
        flow, dec = bundle.sparse_structure_models()
        mods = [bundle.dino, flow, dec]
        bundle.on_device(mods, device)
        try:
            with torch.random.fork_rng(devices=[device]):
                torch.manual_seed(seed)
                cond = encode_images(bundle, [img], device)
                occ = sample_sparse_structure(bundle, cond, steps, cfg, (0.5, 1.0), 3.0, device)
        finally:
            if lato2_model["offload"]:
                bundle.offload(mods)
        if occ.shape[-1] != bundle.min_res:
            raise ValueError(f"LATO2: SS decoder resolution {occ.shape[-1]} != {bundle.min_res}")
        coords = geo.zup_to_yup_coords(geo.dense_to_coords(occ[0].cpu()), bundle.min_res)
        if coords.shape[0] == 0:
            raise RuntimeError("LATO2: empty sparse structure; try another seed.")
        return IO.NodeOutput(Scaffold(coords, bundle.min_res, img, source="image"),
                             geo.np_uint8_to_comfy_image([img]))


class LATO2ScaffoldPreviewMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2ScaffoldPreviewMesh",
            display_name="LATO2 Scaffold Preview Mesh",
            category="3d/LATO2",
            description="Turns the scaffold into a blocky cube mesh, to check voxelization and orientation.",
            inputs=[LATO2Scaffold.Input("scaffold"),
                    IO.Boolean.Input("restore_scale", default=True)],
            outputs=[IO.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, scaffold, restore_scale):
        c = scaffold.voxels.float()
        r = scaffold.resolution
        corners = torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=torch.float32)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        tris = torch.tensor([[a, b, cc] for q in quads for a, b, cc in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
        verts = ((c[:, None, :] + corners[None]) / r - 0.5).reshape(-1, 3)
        faces = (tris[None] + 8 * torch.arange(c.shape[0])[:, None, None]).reshape(-1, 3)
        if restore_scale:
            verts = verts * scaffold.extent + torch.from_numpy(scaffold.center).float()
        return IO.NodeOutput(Types.MESH(vertices=verts[None], faces=faces[None].int()))


# ============================================================================= generation

def _fix_winding(verts, faces):
    import trimesh
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_winding(tm)
    try:
        if tm.is_volume or tm.is_watertight:
            trimesh.repair.fix_inversion(tm)
        else:
            # open mesh: point normals away from the centroid on average
            fc = tm.triangles_center - tm.centroid
            if float((tm.face_normals * fc).sum()) < 0:
                tm.invert()
    except Exception:
        pass
    return np.asarray(tm.vertices, dtype=np.float32), np.asarray(tm.faces, dtype=np.int64)


class LATO2Generate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LATO2Generate",
            display_name="LATO2 Generate Mesh",
            category="3d/LATO2",
            description="V-Flow generates vertex positions at a controllable count, T-Flow predicts their "
                        "connectivity. Output is a clean, artist-style low/mid-poly MESH.",
            inputs=[
                LATO2Model.Input("lato2_model"),
                LATO2Scaffold.Input("scaffold"),
                IO.Int.Input("vertex_count", default=2000, min=200, max=5000, step=50,
                             tooltip="Target vertex count (trained range 200-5000)."),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                IO.Int.Input("vflow_steps", default=24, min=1, max=200),
                IO.Float.Input("vflow_cfg", default=3.0, min=0.0, max=20.0, step=0.1),
                IO.Int.Input("tflow_steps", default=50, min=1, max=500),
                IO.Boolean.Input("fill_open_rings", default=True,
                                 tooltip="Triangulate small holes (chordless open-edge rings) after decoding."),
                IO.Int.Input("max_ring_size", default=12, min=4, max=64),
                IO.Boolean.Input("fix_normals", default=True,
                                 tooltip="Make face winding consistent and outward (LATO.2 faces come unoriented)."),
                IO.Boolean.Input("restore_scale", default=True,
                                 tooltip="Map the result back to the source mesh's position/size (mesh scaffolds), "
                                         "so it lines up with the original for baking."),
                IO.Image.Input("image", optional=True,
                               tooltip="Conditioning image. Overrides the scaffold's own render / crop. Required for "
                                       "voxel scaffolds."),
                IO.Mask.Input("mask", optional=True, tooltip="Foreground mask for `image` (1 = object)."),
                IO.Float.Input("inference_threshold", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                IO.Float.Input("edge_threshold", default=0.0, min=-10.0, max=10.0, step=0.1, advanced=True),
                IO.Float.Input("rescale_t", default=1.0, min=0.1, max=10.0, step=0.1, advanced=True),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Image.Output(display_name="cond_image"),
                IO.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, lato2_model, scaffold, vertex_count, seed, vflow_steps, vflow_cfg, tflow_steps,
                fill_open_rings, max_ring_size, fix_normals, restore_scale, image=None, mask=None,
                inference_threshold=0.5, edge_threshold=0.0, rescale_t=1.0):
        bundle: LATO2Bundle = lato2_model["bundle"]
        offload = lato2_model["offload"]
        device = mm.get_torch_device()

        if image is not None:
            cond_img = _cond_from_image(image, mask, 1.2, bundle.img_res)
        elif scaffold.cond_image is not None:
            cond_img = scaffold.cond_image
        else:
            raise ValueError("LATO2: this scaffold has no conditioning image; connect `image` (and `mask`).")
        if scaffold.resolution != bundle.min_res:
            raise ValueError(f"LATO2: scaffold resolution {scaffold.resolution} != model {bundle.min_res}")

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            bundle.on_device([bundle.dino], device)
            try:
                cond = encode_images(bundle, [cond_img], device)
            finally:
                if offload:
                    bundle.offload([bundle.dino])
            verts, faces, n = generate_mesh(
                bundle, scaffold.voxels, cond, device, vertex_count, vflow_steps, vflow_cfg, rescale_t,
                tflow_steps, inference_threshold, edge_threshold, fill_open_rings, max_ring_size, offload=offload,
            )
        if faces.shape[0] == 0:
            raise RuntimeError(f"LATO2: T-Flow decoded no faces for {n} vertices; try another seed.")
        if fix_normals:
            verts, faces = _fix_winding(verts, faces)
        if restore_scale:
            verts = (verts.astype(np.float64) * scaffold.extent + scaffold.center).astype(np.float32)
        info = f"LATO2: {n} vertices, {faces.shape[0]} faces (target {vertex_count}, seed {seed})"
        log.info(info)
        mesh = Types.MESH(vertices=torch.from_numpy(verts)[None], faces=torch.from_numpy(faces).int()[None])
        return IO.NodeOutput(mesh, geo.np_uint8_to_comfy_image([cond_img]), info)


class LATO2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [LATO2LoadModel, LATO2MeshToScaffold, LATO2VoxelToScaffold, LATO2ImageToScaffold,
                LATO2ScaffoldPreviewMesh, LATO2Generate]


register_folder()


async def comfy_entrypoint() -> LATO2Extension:
    return LATO2Extension()
