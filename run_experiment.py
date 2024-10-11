import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import numpy as np

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, UniPCMultistepScheduler
from diffusers.utils import (
    numpy_to_pil,
)
from PIL import Image

from src.pipeline import StableSyncMVDPipeline
from src.configs import *
from shutil import copy


opt = parse_config()
# print(opt)
override_condition_type = 'depth' # set to False to use the condition type from the config file

if opt.mesh_config_relative:
	mesh_path = join(dirname(opt.config), opt.mesh)
else:
	mesh_path = abspath(opt.mesh)

if opt.output:
	output_root = abspath(opt.output)
else:
	output_root = dirname(opt.config)

output_name_components = []
if opt.prefix and opt.prefix != "":
	output_name_components.append(opt.prefix)
if opt.use_mesh_name:
	mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
	output_name_components.append(mesh_name)

if opt.timeformat and opt.timeformat != "":
	output_name_components.append(datetime.now().strftime(opt.timeformat))
output_name = "_".join(output_name_components)
output_dir = join(output_root, output_name)

if not isdir(output_dir):
	os.mkdir(output_dir)
else:
	print(f"Results exist in the output directory, use time string to avoid name collision.")
	exit(0)

print(f"Saving to {output_dir}")

copy(opt.config, join(output_dir, "config.yaml"))

logging_config = {
	"output_dir":output_dir, 
	# "output_name":None, 
	# "intermediate":False, 
	"log_interval":opt.log_interval,
	"view_fast_preview": opt.view_fast_preview,
	"tex_fast_preview": opt.tex_fast_preview,
	}

if override_condition_type:
    opt.cond_type = override_condition_type

if opt.cond_type == "normal":
	controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
elif opt.cond_type == "depth":
	controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)			

pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)


pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

syncmvd = StableSyncMVDPipeline(**pipe.components)

total_textured_views = []

hits = [1] if opt.max_hits == 1 else list(set([1, opt.max_hits]))

print(f"Max hits: {opt.max_hits}, Hits: {hits}")

for max_hit in hits:
    print(f"Running with max_hit={max_hit}")
    result_tex_rgb, textured_views, v = syncmvd(
        prompt=opt.prompt,
        height=opt.latent_view_size*8,
        width=opt.latent_view_size*8,
        num_inference_steps=opt.steps,
        guidance_scale=opt.guidance_scale,
        negative_prompt=opt.negative_prompt,
        
        generator=torch.manual_seed(opt.seed),
        max_batch_size=64,
        controlnet_guess_mode=opt.guess_mode,
        controlnet_conditioning_scale = opt.conditioning_scale,
        controlnet_conditioning_end_scale= opt.conditioning_scale_end,
        control_guidance_start= opt.control_guidance_start,
        control_guidance_end = opt.control_guidance_end,
        guidance_rescale = opt.guidance_rescale,
        use_directional_prompt=True,

        mesh_path=mesh_path,
        mesh_transform={"scale":opt.mesh_scale},
        mesh_autouv=not opt.keep_mesh_uv,

        camera_azims=opt.camera_azims,
        top_cameras=not opt.no_top_cameras,
        texture_size=opt.latent_tex_size,
        render_rgb_size=opt.rgb_view_size,
        texture_rgb_size=opt.rgb_tex_size,
        multiview_diffusion_end=opt.mvd_end,
        exp_start=opt.mvd_exp_start,
        exp_end=opt.mvd_exp_end,
        ref_attention_end=opt.ref_attention_end,
        shuffle_background_change=opt.shuffle_bg_change,
        shuffle_background_end=opt.shuffle_bg_end,

        logging_config=logging_config,
        cond_type=opt.cond_type,
        max_hits=max_hit,
        style_prompt=opt.style_prompt
    )
    
    textured_views = [textured_view.permute(1,2,0).cpu().numpy() for textured_view in textured_views]
    total_textured_views.append(textured_views)



# Create Sigal Views
output_dir = logging_config["output_dir"]
result_dir = f"{output_dir}/results"
textured_views_save_path = f"{result_dir}/textured_views"
sigal_views_save_path = f"{result_dir}/sigal_views"
os.makedirs(sigal_views_save_path, exist_ok=True)

for i in range(len(total_textured_views[0])):
    # Start with the base image from the first list
    base_image = total_textured_views[0][i]  # Single image from first set

    # Find corresponding images in subsequent lists based on `max_hit` relationships
    concat_images = [base_image]  # Store images to concatenate
    for hit_idx in range(1, len(total_textured_views)):
        # Map each base image to its corresponding images in the higher `max_hit` sets
        corresponding_images = total_textured_views[hit_idx][i * (hit_idx + 1): (i + 1) * (hit_idx + 1)]
        concat_images.extend(corresponding_images)

    # Concatenate all corresponding images vertically
    concat_images_np = np.concatenate(concat_images, axis=0)  # Concatenate along the height (vertical)
    
    img = numpy_to_pil(concat_images_np)[0]

    # Check if the image is in RGBA mode and convert to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Save the image in JPEG format
    img.save(os.path.join(sigal_views_save_path, f"concatenated_view_{i:02d}.jpg"), format="JPEG")

display(v)