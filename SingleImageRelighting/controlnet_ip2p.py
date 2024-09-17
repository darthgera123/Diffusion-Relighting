import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "lllyasviel/control_v11e_sd15_ip2p"

image = Image.open('diffuse.png')
prompt = "reconstruct"

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                             cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "benjamin-paine/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
    cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=image).images[0]

image.save('transform.png')