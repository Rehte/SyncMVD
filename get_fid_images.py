import numpy as np

import torch
from trimesh import Trimesh

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    AmbientLights,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesUV
)


from src.renderer.project import RaycastingImaging
from src.renderer.shader import HardNChannelFlatShader

def get_uv_map():
    pass

# Set the cameras given the camera poses and centers
def set_cameras(self, camera_poses, centers=None, camera_distance=4.0, scale=None):
    elev = torch.FloatTensor([pose[0] for pose in camera_poses])
    azim = torch.FloatTensor([pose[1] for pose in camera_poses])
    R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
    self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))

def get_inner_hit_plane(
    mesh,
    cameras, 
    max_hits=2,
    remove_backface_hits=True,
    target_size=(1024, 1024),
    channels=3,
    device='cuda',
    sampling_mode='nearest',
    size=1024,
    blur=0.0, 
    face_per_pix=1, 
    perspective_correct=False
    ):
    lights = AmbientLights(ambient_color=((1.0,)*channels,), device=device)
    raster_settings = RasterizationSettings(
        image_size=size, 
        blur_radius=blur, 
        faces_per_pixel=face_per_pix,
        perspective_correct=perspective_correct,
        cull_backfaces=False,
        max_faces_per_bin=30000,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,

        ),
        shader=HardNChannelFlatShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            channels=channels
            # materials=materials
        )
    )
    
    vertices = mesh.verts_packed().cpu().numpy()  # (V, 3) shape, move to CPU and convert to numpy
    faces = mesh.faces_packed().cpu().numpy()  # (F, 3) shape, move to CPU and convert to numpy

    raycast = RaycastingImaging()

    visible_faces_list = []
    visible_texture_map_list = []
    mesh_face_indices_list = []
    
    for k, camera in enumerate(cameras):
        R = camera.R.cpu().numpy()
        T = camera.T.cpu().numpy()

        Rt = np.eye(4)  # Start with an identity matrix
        Rt[:3, :3] = np.swapaxes(R, 1, 2)  # Top-left 3x3 is the transposed rotation
        Rt[:3, 3] = T   # Top-right 3x1 is the inverted translation

        mesh_frame = Trimesh(vertices=vertices, faces=faces).apply_transform(Rt)
        # mesh_frame.export(str(k)+"trans.ply")

        c2w = np.eye(4).astype(np.float32)[:3]
        raycast.prepare(image_height=512 * 3, image_width=512 * 3, c2w=c2w)
        ray_indexes, points, mesh_face_indices = raycast.get_image(mesh_frame, max_hits * 2 - 1)   
        
        for i in range(max_hits):
            # mesh_face_indexes = np.hstack([mesh_face_indices[i], np.array([mesh_face_indices[i][-1] for _ in range(faces.shape[0] - mesh_face_indices[i].shape[0])])])
            idx = i * 2 if remove_backface_hits else i
            visible_faces = faces[mesh_face_indices[idx]]  # Only keep the visible faces
            mesh_face_indices_list.append(torch.tensor(mesh_face_indices[idx], dtype=torch.int64, device='cuda'))
            # Trimesh(vertices=vertices, faces=visible_faces).export(str(k)+"trans"+str(i)+".ply")
            visible_faces = torch.tensor(visible_faces, dtype=torch.int64, device='cuda')

            visible_faces_list.append(visible_faces)
            new_map = torch.zeros(target_size+(channels,), device=device)
            visible_texture_map_list.append(mesh.textures.faces_uvs_padded()[0, mesh_face_indices[idx]])
    
    textures = TexturesUV(
        [new_map] * len(cameras) * max_hits, 
        visible_texture_map_list, 
        [mesh.textures.verts_uvs_padded()[0]] * len(cameras) * max_hits, 
        sampling_mode=sampling_mode
    )
    occ_mesh = Meshes(verts = [mesh.verts_packed()] * len(cameras) * max_hits, faces = visible_faces_list, textures = textures)
    occ_cameras = FoVOrthographicCameras(device=device, R=cameras.R.repeat_interleave(max_hits, 0), T=cameras.T.repeat_interleave(max_hits, 0), scale_xyz=cameras.scale_xyz.repeat_interleave(max_hits, 0))
    images_predicted = renderer(occ_mesh, cameras=occ_cameras, lights=lights)

def main():
    pass

if __name__ == "__main__":
    main()