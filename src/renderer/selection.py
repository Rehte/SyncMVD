import random
from PIL import Image
import os
from tqdm import tqdm
from typing import NamedTuple, Sequence
import numpy as np
import cv2

import torch
from torchvision import transforms
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    PerspectiveCameras,
    look_at_view_transform,
    AmbientLights,
    SoftPhongShader,
    TexturesUV,
)

QUAD_WEIGHTS = {
    0: 0, # background
    1: 0.1,   # old
    2: 0.5,  # update
    3: 1    # new
}

PALETTE = {
    0: [255, 255, 255], # white  -  background
    1: [204, 50, 50],   # red    -  old
    2: [231, 180, 22],  # yellow -  update
    3: [45, 201, 55]    # green  -  new
}


class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)

class FlatTexelShader(ShaderBase):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        return texels.squeeze(-2)

def visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_score, device):
    quad_mask_tensor = quad_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    quad_mask_image_tensor = torch.zeros_like(quad_mask_tensor)
    
    for idx in PALETTE:
        selected = quad_mask_tensor[quad_mask_tensor == idx].reshape(-1, 3)
        selected = torch.FloatTensor(PALETTE[idx]).to(device).unsqueeze(0).repeat(selected.shape[0], 1)

        quad_mask_image_tensor[quad_mask_tensor == idx] = selected.reshape(-1)

    quad_mask_image_np = quad_mask_image_tensor[0].cpu().numpy().astype(np.uint8)
    quad_mask_image = Image.fromarray(quad_mask_image_np).convert("RGB")
    quad_mask_image.save(os.path.join(mask_image_dir, "{}_quad_{:.5f}.png".format(view_idx, view_score)))

def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[update_mask_tensor == 1] = 2
    quad_mask_tensor[new_mask_tensor == 1] = 3

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor

@torch.no_grad()
def build_diffusion_mask(mesh_stuff, 
    renderer, exist_texture, similarity_texture_cache, target_value, device, image_size, 
    smooth_mask=False, view_threshold=0.01, textures_idx=None):

    mesh, faces, verts_uvs = mesh_stuff
    mask_mesh = mesh.clone() # NOTE in-place operation - DANGER!!!

    # visible mask => the whole region
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=torch.ones_like(exist_texture_expand),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    # visible_mask_tensor, *_ = render(mask_mesh, renderer)
    visible_mask_tensor, _, similarity_map_tensor, *_ = render(mask_mesh, renderer)
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= view_threshold).float()
    visible_mask_tensor *= valid_mask_tensor

    # nonexist mask <=> new mask
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=1 - exist_texture_expand,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    new_mask_tensor, *_ = render(mask_mesh, renderer)
    new_mask_tensor *= valid_mask_tensor

    # exist mask => visible mask - new mask
    exist_mask_tensor = visible_mask_tensor - new_mask_tensor
    exist_mask_tensor[exist_mask_tensor < 0] = 0 # NOTE dilate can lead to overflow

    # all update mask
    mask_mesh.textures = TexturesUV(
        maps=(
            similarity_texture_cache.argmax(0) == target_value
            # # only consider the views that have already appeared before
            # similarity_texture_cache[0:target_value+1].argmax(0) == target_value
        ).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    all_update_mask_tensor, *_ = render(mask_mesh, renderer)

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor

    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask

@torch.no_grad()
def render(mesh, renderer, pad_value=10):
    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        return pixel_normals

    def similarity_shading(meshes, fragments):
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        vertices = meshes.verts_packed()  # (V, 3)
        face_positions = vertices[faces]
        view_directions = torch.nn.functional.normalize((renderer.shader.cameras.get_camera_center().reshape(1, 1, 3) - face_positions), p=2, dim=2)
        cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals, view_directions)
        pixel_similarity = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, cosine_similarity.unsqueeze(-1)
        )

        return pixel_similarity

    def get_relative_depth_map(fragments, pad_value=pad_value):
        absolute_depth = fragments.zbuf[..., 0] # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black

        return relative_depth


    images, fragments = renderer(mesh)
    normal_maps = phong_normal_shading(mesh, fragments).squeeze(-2)
    similarity_maps = similarity_shading(mesh, fragments).squeeze(-2) # -1 - 1
    depth_maps = get_relative_depth_map(fragments)

    # normalize similarity mask to 0 - 1 
    similarity_maps = torch.abs(similarity_maps) # 0 - 1
    
    # HACK erode, eliminate isolated dots
    non_zero_similarity = (similarity_maps > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_maps.device).unsqueeze(0) / 255.
    similarity_maps = non_zero_similarity.unsqueeze(-1) * similarity_maps 

    return images, normal_maps, similarity_maps, depth_maps, fragments

def init_flat_texel_shader(camera, device):
    shader=FlatTexelShader(
        cameras=camera,
        device=device
    )
    
    return shader

def init_soft_phong_shader(camera, blend_params, device):
    lights = AmbientLights(device=device)
    shader = SoftPhongShader(
        cameras=camera,
        lights=lights,
        device=device,
        blend_params=blend_params
    )

    return shader

def init_camera(dist, elev, azim, image_size, device):
    R, T = look_at_view_transform(dist, elev, azim)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)

    return cameras

def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=shader
    )

    return renderer

@torch.no_grad()
def render_one_view(mesh,
    dist, elev, azim,
    image_size, faces_per_pixel,
    device):

    # render the view
    cameras = init_camera(
        dist, elev, azim,
        image_size, device
    )
    renderer = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size, 
        faces_per_pixel=faces_per_pixel
    )

    init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments = render(mesh, renderer)
    
    return (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    )

def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat

@torch.no_grad()
def render_one_view_and_build_masks(dist, elev, azim, 
    selected_view_idx, view_idx, view_punishments,
    similarity_texture_cache, exist_texture,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, save_intermediate=False, smooth_mask=False, view_threshold=0.01,
    textures_idx=None,
    hit=1
    ):
    
    # render the view
    (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    ) = render_one_view(mesh,
        dist, elev, azim,
        image_size, faces_per_pixel,
        device
    )
    
    init_image = init_images_tensor[0].cpu()
    init_image = init_image.permute(2, 0, 1)
    init_image = transforms.ToPILImage()(init_image).convert("RGB")

    normal_map = normal_maps_tensor[0].cpu()
    normal_map = normal_map.permute(2, 0, 1)
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")

    depth_map = depth_maps_tensor[0].cpu().numpy()
    depth_map = Image.fromarray(depth_map).convert("L")

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")


    flat_renderer = init_renderer(cameras,
        shader=init_flat_texel_shader(
            camera=cameras,
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (mesh, faces, verts_uvs), 
        flat_renderer, exist_texture, similarity_texture_cache, selected_view_idx, device, image_size, 
        smooth_mask=smooth_mask, view_threshold=view_threshold, textures_idx=textures_idx
    )
    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (
        old_mask_tensor, 
        update_mask_tensor, 
        new_mask_tensor, 
        all_mask_tensor, 
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}_{}.png".format(view_idx, hit)))
        normal_map.save(os.path.join(normal_map_dir, "{}_{}.png".format(view_idx, hit)))
        depth_map.save(os.path.join(depth_map_dir, "{}_{}.png".format(view_idx, hit)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}_{}.png".format(view_idx, hit)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_{}_new.png".format(view_idx, hit)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_{}_update.png".format(view_idx, hit)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_{}_old.png".format(view_idx, hit)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_{}_exist.png".format(view_idx, hit)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        renderer, cameras, fragments,
        init_image, normal_map, depth_map, 
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
        old_mask_image, update_mask_image, new_mask_image, 
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor
    )
    
def select_viewpoint(selected_view_ids, view_punishments,
    mode, dist_list, elev_list, azim_list, sector_list, view_idx,
    similarity_texture_cache, exist_texture,
    mesh, faces, verts_uvs,
    image_size, faces_per_pixel,
    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
    device, use_principle=False
):
    if mode == "sequential":
        
        num_views = len(dist_list)

        dist = dist_list[view_idx % num_views]
        elev = elev_list[view_idx % num_views]
        azim = azim_list[view_idx % num_views]
        sector = sector_list[view_idx % num_views]
        
        selected_view_ids.append(view_idx % num_views)

    elif mode == "heuristic":

        if use_principle and view_idx < 6:

            selected_view_idx = view_idx

        else:

            selected_view_idx = None
            max_heat = 0

            print("=> selecting next view...")
            view_heat_list = []
            for sample_idx in tqdm(range(len(dist_list))):

                view_heat, *_ = render_one_view_and_build_masks(dist_list[sample_idx], elev_list[sample_idx], azim_list[sample_idx], 
                    sample_idx, sample_idx, view_punishments,
                    similarity_texture_cache, exist_texture,
                    mesh, faces, verts_uvs,
                    image_size, faces_per_pixel,
                    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                    device)

                if view_heat > max_heat:
                    selected_view_idx = sample_idx
                    max_heat = view_heat

                view_heat_list.append(view_heat.item())

            print(view_heat_list)
            print("select view {} with heat {}".format(selected_view_idx, max_heat))


        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

        view_punishments[selected_view_idx] *= 0.01

    elif mode == "random":

        selected_view_idx = random.choice(range(len(dist_list)))

        dist = dist_list[selected_view_idx]
        elev = elev_list[selected_view_idx]
        azim = azim_list[selected_view_idx]
        sector = sector_list[selected_view_idx]
        
        selected_view_ids.append(selected_view_idx)

    else:
        raise NotImplementedError()

    return dist, elev, azim, sector, selected_view_ids, view_punishments

def init_xatlas():
    pass
    # xatlas mesh