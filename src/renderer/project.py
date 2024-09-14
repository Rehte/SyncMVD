import torch
import pytorch3d
from trimesh.ray.ray_pyembree import RayMeshIntersector
import trimesh
from collections import defaultdict


from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras, 
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

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader
from .voronoi import voronoi_solve

import numpy as np


# Copied from XRay
class RaycastingImaging:
    def __init__(self):
        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

    def __del__(self):
        del self.rays_screen_coords
        del self.rays_origins
        del self.rays_directions

    def prepare(self, image_height, image_width, intrinsics=None, c2w=None):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays((image_height, image_width), intrinsics, c2w)

    def get_image(self, mesh):  #, features):
        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]
        colors = mesh.visual.face_colors[mesh_face_indexes]

        mesh_face_indexes = np.unique(mesh_face_indexes)
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])
        direction = self.rays_directions[ray_indexes][0]
        return ray_indexes, points, normals, colors, direction, mesh_vertex_indexes, mesh_face_indexes

        # assemble mesh fragment into a submesh
        # nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)
        # return ray_indexes, points, normals, nbhood, mesh_vertex_indexes, mesh_face_indexes

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

def generate_rays(image_resolution, intrinsics, c2w):
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T  # [h, w, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)
    
    i, j = grid[..., 1], grid[..., 0]
    directions = np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)], -1) # (H, W, 3)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    
    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
        return_locations=True)
    return index_triangles, index_ray, point_cloud

# Pytorch3D based renderering functions, managed in a class
# Render size is recommended to be the same as your latent view size
# DO NOT USE "bilinear" sampling when you are handling latents.
# Stable Diffusion has 4 latent channels so use channels=4

class UVProjection():
	def __init__(self, texture_size=96, render_size=64, sampling_mode="nearest", channels=3, device=None):
		self.channels = channels
		self.device = device or torch.device("cpu")
		self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
		self.target_size = (texture_size,texture_size)
		self.render_size = render_size
		self.sampling_mode = sampling_mode


	# Load obj mesh, rescale the mesh to fit into the bounding box
	def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		mesh = load_objs_as_meshes([mesh_path], device=self.device)
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			scale = (max_bb - min_bb).max()/2
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))		
		else:
			mesh.scale_verts_((scale_factor))

		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh)
		self.mesh = mesh


	def load_glb_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
		io = IO()
		io.register_meshes_format(MeshGlbFormat())
		with open(mesh_path, "rb") as f:
			mesh = io.load_mesh(f, include_textures=True, device=self.device)
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			scale = (max_bb - min_bb).max()/2 
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))
		else:
			mesh.scale_verts_((scale_factor))
		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh)
		self.mesh = mesh


	# Save obj mesh
	def save_mesh(self, mesh_path, texture):
		save_obj(mesh_path, 
				self.mesh.verts_list()[0],
				self.mesh.faces_list()[0],
				verts_uvs= self.mesh.textures.verts_uvs_list()[0],
				faces_uvs= self.mesh.textures.faces_uvs_list()[0],
				texture_map=texture)

	# Code referred to TEXTure code (https://github.com/TEXTurePaper/TEXTurePaper.git)
	def uv_unwrap(self, mesh):
		verts_list = mesh.verts_list()[0]
		faces_list = mesh.faces_list()[0]


		import xatlas
		import numpy as np
		v_np = verts_list.cpu().numpy()
		f_np = faces_list.int().cpu().numpy()
		atlas = xatlas.Atlas()
		atlas.add_mesh(v_np, f_np)
		chart_options = xatlas.ChartOptions()
		chart_options.max_iterations = 4
		atlas.generate(chart_options=chart_options)
		vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

		vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
		ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)

		new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
		new_tex = TexturesUV(
			[new_map], 
			[ft], 
			[vt], 
			sampling_mode=self.sampling_mode
			)

		mesh.textures = new_tex
		return mesh


	'''
		A functions that disconnect faces in the mesh according to
		its UV seams. The number of vertices are made equal to the
		number of unique vertices its UV layout, while the faces list
		is intact.
	'''
	def disconnect_faces(self):
		mesh = self.mesh
		verts_list = mesh.verts_list()
		faces_list = mesh.faces_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		faces_uvs_list = mesh.textures.faces_uvs_list()
		packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
		verts_disconnect_list = [
			torch.zeros(
				(verts_uvs_list[i].shape[0], 3), 
				dtype=verts_list[0].dtype, 
				device=verts_list[0].device
			) 
			for i in range(len(verts_list))]
		for i in range(len(verts_list)):
			verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
		assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
		self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
		return self.mesh_d


	'''
		A function that construct a temp mesh for back-projection.
		Take a disconnected mesh and a rasterizer, the function calculates
		the projected faces as the UV, as use its original UV with pseudo
		z value as world space geometry.
	'''
	def construct_uv_mesh(self):
		mesh = self.mesh_d
		verts_list = mesh.verts_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		# faces_list = [torch.flip(faces, [-1]) for faces in mesh.faces_list()]
		new_verts_list = []
		for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
			verts = verts.clone()
			verts_uv = verts_uv.clone()
			verts[...,0:2] = verts_uv[...,:]
			verts = (verts - 0.5) * 2
			verts[...,2] *= 1
			new_verts_list.append(verts)
		textures_uv = mesh.textures.clone()
		self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
		return self.mesh_uv


	# Set texture for the current mesh.
	def set_texture_map(self, texture):
		new_map = texture.permute(1, 2, 0)
		new_map = new_map.to(self.device)
		new_tex = TexturesUV(
			[new_map], 
			self.mesh.textures.faces_uvs_padded(), 
			self.mesh.textures.verts_uvs_padded(), 
			sampling_mode=self.sampling_mode
			)
		self.mesh.textures = new_tex


	# Set the initial normal noise texture
	# No generator here for replication of the experiment result. Add one as you wish
	def set_noise_texture(self, channels=None):
		if not channels:
			channels = self.channels
		noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
		self.set_texture_map(noise_texture)
		return noise_texture


	# Set the cameras given the camera poses and centers
	def set_cameras(self, camera_poses, centers=None, camera_distance=2.7, scale=None):
		elev = torch.FloatTensor([pose[0] for pose in camera_poses])
		azim = torch.FloatTensor([pose[1] for pose in camera_poses])
		R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
		self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))


	# Set all necessary internal data for rendering and texture baking
	# Can be used to refresh after changing camera positions
	def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.7, render_size=None, scale=None):
		self.set_cameras(camera_poses, centers, camera_distance, scale=scale)
		if render_size is None:
			render_size = self.render_size
		if not hasattr(self, "renderer"):
			self.setup_renderer(size=render_size)
		if not hasattr(self, "mesh_d"):
			self.disconnect_faces()
		if not hasattr(self, "mesh_uv"):
			self.construct_uv_mesh()
		self.calculate_tex_gradient()
		self.calculate_visible_triangle_mask()
		_,_,_,cos_maps,_, _ = self.render_geometry()
		self.calculate_cos_angle_weights(cos_maps)


	# Setup renderers for rendering
	# max faces per bin set to 30000 to avoid overflow in many test cases.
	# You can use default value to let pytorch3d handle that for you.
	def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
		if not channels:
			channels = self.channels

		self.raster_settings = RasterizationSettings(
			image_size=size, 
			blur_radius=blur, 
			faces_per_pixel=face_per_pix,
			perspective_correct=perspective_correct,
			cull_backfaces=True,
			max_faces_per_bin=30000,
		)

		self.renderer = MeshRenderer(
			rasterizer=MeshRasterizer(
				cameras=self.cameras, 
				raster_settings=self.raster_settings,

			),
			shader=HardNChannelFlatShader(
				device=self.device, 
				cameras=self.cameras,
				lights=self.lights,
				channels=channels
				# materials=materials
			)
		)


	# Bake screen-space cosine weights to UV space
	# May be able to reimplement using the generic "bake_texture" function, but it works so leave it here for now
	@torch.enable_grad()
	def calculate_cos_angle_weights(self, cos_angles, fill=True, channels=None):
		if not channels:
			channels = self.channels
		cos_maps = []
		tmp_mesh = self.mesh.clone()
		for i in range(len(self.cameras)):
			
			zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
			optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
			optimizer.zero_grad()
			zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = zero_tex

			images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)

			loss = torch.sum((cos_angles[i,:,:,0:1]**1 - images_predicted)**2)
			loss.backward()
			optimizer.step()

			if fill:
				zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
				zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,0])
			else:
				zero_map = zero_map.detach() / (self.gradient_maps[i]+1E-8)
			cos_maps.append(zero_map)
		self.cos_maps = cos_maps

		
	# Get geometric info from fragment shader
	# Can be used for generating conditioning image and cosine weights
	# Returns some information you may not need, remember to release them for memory saving
	@torch.no_grad()
	def render_geometry(self, image_size=None):
		if image_size:
			size = self.renderer.rasterizer.raster_settings.image_size
			self.renderer.rasterizer.raster_settings.image_size = image_size
		shader = self.renderer.shader
		self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights)
		tmp_mesh = self.mesh.clone()

		raycast = RaycastingImaging()
		
		for camera in (self.cameras):
			# c2w = np.array(frame["c2w"])
			# mesh.export("gt.ply")
			# Rt = np.linalg.inv(c2w)

			# is this right?
			Rt = np.matmul(np.array(camera.R), np.array(camera.T))
			invRt = np.linalg.inv(Rt)

			vertices = tmp_mesh.verts_packed().cpu().numpy()  # (V, 3) shape, move to CPU and convert to numpy
			faces = tmp_mesh.faces_packed().cpu().numpy()  # (F, 3) shape, move to CPU and convert to numpy

			mesh_frame = trimesh.Trimesh(vertices=vertices, faces=faces).apply_transform(Rt)
			
			# mesh_frame.export("trans.ply")
			c2w = np.eye(4).astype(np.float32)[:3]

			camera_angle_x = camera.fov
			focal_length = 0.5 * size[0] / np.tan(0.5 * camera_angle_x)

			cx = size[0] / 2.0
			cy = size[1] / 2.0

			intrinsics = np.array([[focal_length, 0, cx],
								[0, focal_length, cy],
								[0, 0, 1]])
			
			raycast.prepare(image_height=size[1], image_width=size[0], intrinsics=intrinsics, c2w=c2w)
			
			ray_indexes, points, normals, colors, direction, mesh_vertex_indexes, mesh_face_indexes = raycast.get_image(mesh_frame)   
			
			# normalize normals
			normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
			colors = colors[:, :3] / 255.0

			# collect points and normals for each ray
			ray_points = defaultdict(list)
			ray_normals = defaultdict(list)
			ray_colors = defaultdict(list)
			for ray_index, point, normal, color in zip(ray_indexes, points, normals, colors):
				ray_points[ray_index].append(point)
				ray_normals[ray_index].append(normal)
				ray_colors[ray_index].append(color)

			# ray to image
			max_hits = 4
			GenDepths = np.zeros((max_hits, 1+3+3, size[0], size[1]), dtype=np.float32)

			for i in range(max_hits):
				for ray_index, ray_point in ray_points.items():
					if i < len(ray_point):
						u = ray_index // size[1]
						v = ray_index % size[1]
						np.linalg.norm(ray_point[i] - c2w[:, 3])
						np.linalg.norm(ray_normals[ray_index][i])
						cos_angle = np.clip(np.dot(np.linalg.norm(ray_point[i] - c2w[:, 3]), np.linalg.norm(ray_normals[ray_index][i])), 0, 1)

						GenDepths[i, 0:3, u, v] = np.dot(invRt, np.append(ray_point[i] - c2w[:, 3], 1))[:3]
						GenDepths[i, 3:6, u, v] = np.dot(invRt, np.append(ray_normals[ray_index][i], 0))[:3]
						GenDepths[i, 6, u, v] = np.linalg.norm(ray_point[i] - c2w[:, 3])
						GenDepths[i, 7, u, v] = cos_angle
						# GenDepths[i, 4:7, u, v] = ray_colors[ray_index][i]
		
		# TODO: Implement XRay, How do I get fXXXing cos_angles?
		# 09/04 : I guess depth and cos_angle would be needed
		verts, normals, depths, cos_angles, texels, fragments = self.renderer(tmp_mesh.extend(len(self.cameras)), cameras=self.cameras, lights=self.lights)
		self.renderer.shader = shader

		if image_size:
			self.renderer.rasterizer.raster_settings.image_size = size

		return verts, normals, depths, cos_angles, texels, fragments


	# Project world normal to view space and normalize
	@torch.no_grad()
	def decode_view_normal(self, normals):
		w2v_mat = self.cameras.get_full_projection_transform()
		normals_view = torch.clone(normals)[:,:,:,0:3]
		normals_view = normals_view.reshape(normals_view.shape[0], -1, 3)
		normals_view = w2v_mat.transform_normals(normals_view)
		normals_view = normals_view.reshape(normals.shape[0:3]+(3,))
		normals_view[:,:,:,2] *= -1
		normals = (normals_view[...,0:3]+1) * normals[...,3:] / 2 + torch.FloatTensor(((((0.5,0.5,1))))).to(self.device) * (1 - normals[...,3:])
		# normals = torch.cat([normal for normal in normals], dim=1)
		normals = normals.clamp(0, 1)
		return normals


	# Normalize absolute depth to inverse depth
	@torch.no_grad()
	def decode_normalized_depth(self, depths, batched_norm=False):
		view_z, mask = depths.unbind(-1)
		view_z = view_z * mask + 100 * (1-mask)
		inv_z = 1 / view_z
		inv_z_min = inv_z * mask + 100 * (1-mask)
		if not batched_norm:
			max_ = torch.max(inv_z, 1, keepdim=True)
			max_ = torch.max(max_[0], 2, keepdim=True)[0]

			min_ = torch.min(inv_z_min, 1, keepdim=True)
			min_ = torch.min(min_[0], 2, keepdim=True)[0]
		else:
			max_ = torch.max(inv_z)
			min_ = torch.min(inv_z_min)
		inv_z = (inv_z - min_) / (max_ - min_)
		inv_z = inv_z.clamp(0,1)
		inv_z = inv_z[...,None].repeat(1,1,1,3)

		return inv_z


	# Multiple screen pixels could pass gradient to a same texel
	# We can precalculate this gradient strength and use it to normalize gradients when we bake textures
	@torch.enable_grad()
	def calculate_tex_gradient(self, channels=None):
		if not channels:
			channels = self.channels
		tmp_mesh = self.mesh.clone()
		gradient_maps = []
		for i in range(len(self.cameras)):
			zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
			optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
			optimizer.zero_grad()
			zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = zero_tex
			images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)
			loss = torch.sum((1 - images_predicted)**2)
			loss.backward()
			optimizer.step()

			gradient_maps.append(zero_map.detach())

		self.gradient_maps = gradient_maps


	# Get the UV space masks of triangles visible in each view
	# First get face ids from each view, then filter pixels on UV space to generate masks
	@torch.no_grad()
	def calculate_visible_triangle_mask(self, channels=None, image_size=(512,512)):
		if not channels:
			channels = self.channels

		pix2face_list = []
		for i in range(len(self.cameras)):
			self.renderer.rasterizer.raster_settings.image_size=image_size
			pix2face = self.renderer.rasterizer(self.mesh_d, cameras=self.cameras[i]).pix_to_face
			self.renderer.rasterizer.raster_settings.image_size=self.render_size
			pix2face_list.append(pix2face)

		if not hasattr(self, "mesh_uv"):
			self.construct_uv_mesh()

		raster_settings = RasterizationSettings(
			image_size=self.target_size, 
			blur_radius=0, 
			faces_per_pixel=1,
			perspective_correct=False,
			cull_backfaces=False,
			max_faces_per_bin=30000,
			)

		R, T = look_at_view_transform(dist=2, elev=0, azim=0)
		cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

		rasterizer=MeshRasterizer(
			cameras=cameras, 
			raster_settings=raster_settings
		)
		uv_pix2face = rasterizer(self.mesh_uv).pix_to_face

		visible_triangles = []
		for i in range(len(pix2face_list)):
			valid_faceid = torch.unique(pix2face_list[i])
			valid_faceid = valid_faceid[1:] if valid_faceid[0]==-1 else valid_faceid
			mask = torch.isin(uv_pix2face[0], valid_faceid, assume_unique=False)
			# uv_pix2face[0][~mask] = -1
			triangle_mask = torch.ones(self.target_size+(1,), device=self.device)
			triangle_mask[~mask] = 0
			
			triangle_mask[:,1:][triangle_mask[:,:-1] > 0] = 1
			triangle_mask[:,:-1][triangle_mask[:,1:] > 0] = 1
			triangle_mask[1:,:][triangle_mask[:-1,:] > 0] = 1
			triangle_mask[:-1,:][triangle_mask[1:,:] > 0] = 1
			visible_triangles.append(triangle_mask)

		self.visible_triangles = visible_triangles



	# Render the current mesh and texture from current cameras
	def render_textured_views(self):
		meshes = self.mesh.extend(len(self.cameras))
		images_predicted = self.renderer(meshes, cameras=self.cameras, lights=self.lights)

		return [image.permute(2, 0, 1) for image in images_predicted]


	# Bake views into a texture
	# First bake into individual textures then combine based on cosine weight
	@torch.enable_grad()
	def bake_texture(self, views=None, main_views=[], cos_weighted=True, channels=None, exp=None, noisy=False, generator=None):
		# TODO: Implement texture baking w/ occluded region
		if not exp:
			exp=1
		if not channels:
			channels = self.channels
		views = [view.permute(1, 2, 0) for view in views]

		tmp_mesh = self.mesh
		bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
		optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
		optimizer.zero_grad()
		loss = 0
		for i in range(len(self.cameras)):    
			bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
			tmp_mesh.textures = bake_tex
			images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights, device=self.device)
			predicted_rgb = images_predicted[..., :-1]
			loss += (((predicted_rgb[...] - views[i]))**2).sum()
		loss.backward(retain_graph=False)
		optimizer.step()

		total_weights = 0
		baked = 0
		for i in range(len(bake_maps)):
			normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[i] + 1E-8)
			bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[i][...,0])
			weight = self.visible_triangles[i] * (self.cos_maps[i]) ** exp
			if noisy:
				noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
				weight *= noise
			total_weights += weight
			baked += bake_map * weight
		baked /= total_weights + 1E-8
		baked = voronoi_solve(baked, total_weights[...,0])

		bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
		tmp_mesh.textures = bake_tex
		extended_mesh = tmp_mesh.extend(len(self.cameras))
		images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
		learned_views = [image.permute(2, 0, 1) for image in images_predicted]

		return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)


	# Move the internel data to a specific device
	def to(self, device):
		for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
			if hasattr(self, mesh_name):
				mesh = getattr(self, mesh_name)
				setattr(self, mesh_name, mesh.to(device))
		for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
			if hasattr(self, list_name):
				map_list = getattr(self, list_name)
				for i in range(len(map_list)):
					map_list[i] = map_list[i].to(device)
