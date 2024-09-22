import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
	look_at_view_transform,
    FoVOrthographicCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
	MeshRasterizer,
)
from diffusers.utils import (
    numpy_to_pil,
)

from src.renderer.shader import HardNChannelFlatShader

def render_textured_views(mesh, cameras, lights, renderer):
    images_predicted = []
    for i, camera in enumerate(cameras):
        image = renderer(mesh, cameras=camera, lights=lights)
        images_predicted.append(image[0].permute(2, 0, 1))

    return images_predicted

def set_cameras(camera_poses, centers=None, camera_distance=2.7, scale=None):
    elev = torch.FloatTensor([pose[0] for pose in camera_poses])
    azim = torch.FloatTensor([pose[1] for pose in camera_poses])
    R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
    return FoVOrthographicCameras(device='cuda', R=R, T=T, scale_xyz=scale or ((1,1,1),))

def set_camera_poses():
    # Define the cameras for rendering
    camera_poses = []
    
    camera_azims = [-180, -135, -90, -45, 0, 45, 90, 135]

    for i, azim in enumerate(camera_azims):
        if azim < 0:
            azim += 360
        camera_poses.append((0, azim))

    # Add two additional cameras for painting the top surfaces
    camera_poses.append((30, 0))
    camera_poses.append((30, 180))
    
    return camera_poses

def set_lights():
    return AmbientLights(ambient_color=((1.0,)*3,), device='cuda')

def set_renderer(cameras, lights):
    raster_settings = RasterizationSettings(
        image_size=(1024, 1024), 
        blur_radius=0, 
        faces_per_pixel=1,
        perspective_correct=False,
        cull_backfaces=False,
        max_faces_per_bin=30000,
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=HardNChannelFlatShader(
            device='cuda',
            cameras=cameras,
            lights=lights,
            channels=3
            # materials=materials
        )
    )
    return renderer

def save_views(rendered_views):
    textured_views_rgb = torch.cat(rendered_views, axis=-1)[:-1,...]
    textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
    v = numpy_to_pil(textured_views_rgb)[0]
    v.save(f"output/textured_views_rgb.jpg")

def main():
    obj_path = "output/textured.obj"
    
    mesh = load_objs_as_meshes([obj_path], device='cuda')
    
    verts = mesh.verts_packed()
    max_bb = (verts - 0).max(0)[0]
    min_bb = (verts - 0).min(0)[0]
    scale = (max_bb - min_bb).max()/2
    center = (max_bb+min_bb) /2
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1 / float(scale)))	
    
    lights = set_lights()
    camera_poses = set_camera_poses()
    cameras = set_cameras(camera_poses)
    renderer = set_renderer(cameras, lights)
    rendered_views = render_textured_views(mesh, cameras, lights, renderer)
    save_views(rendered_views)
    

if __name__ == '__main__':
    main()