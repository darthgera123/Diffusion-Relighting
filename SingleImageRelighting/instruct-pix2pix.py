import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--prompt",default="turn her into cyborg")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, 
                                                                cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # `image` is an RGB PIL.Image
    image = PIL.Image.open('relit.png')
    images = pipe(args.prompt, image=image).images
    gen_img = images[0]
    gen_img.save('transform.png')