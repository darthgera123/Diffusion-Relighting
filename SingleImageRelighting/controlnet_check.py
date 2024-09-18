import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
)
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from cunet import EnvMapEncoder
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class CustomControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels = self.block_out_channels[0],
            block_out_channels=self.conditioning_embedding_out_channels,
            conditioning_channels=320,
        )

class ControlNetTest(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding



class CustomEnvMapControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.controlnet_cond_embedding = EnvMapEncoder(map_size=64,latent_size=320)
        self.controlnet_cond_embedding.additional_layers = zero_module(
            self.controlnet_cond_embedding.add
        )

        



def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--relit_path",type=str,default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--env_path",type=str,default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/envmaps/med_exr")
    parser.add_argument("--diffuse_path",type=str,default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--mask_path",type=str,default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--normal_path",type=str,default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--tokenizer_name",type=str,default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--revision",type=str,default=None,required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None,required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    args = parser.parse_args()
    return args

# class CustomControlNet(ControlNetModel):
#     def __init__(self, unet_config):
#         super().__init__(unet_config)
#         # Replace the controlnet conditioning embedding with a custom one
#         self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
#             in_channels=320,  # Your control signal's channels
#             out_channels=unet_config.conv_in.out_channels,  # Match UNet's expected channels
#             conditioning_channels=unet_config.conv_in.out_channels,
#             num_channels=unet_config.conv_in.out_channels,
#             num_groups=32,
#             activation=unet_config.activation_fn,
#         )


if __name__ == "__main__":
    
    args = parse_args()

    num_classes = 6
    batch_size = 4
    light = torch.rand(4,3,256,512)

    
    image = torch.rand(4,3,512,512)
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, \
        subfolder="unet", revision=args.revision, variant=args.variant,\
        cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, \
                                        subfolder="vae",\
                                        cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    
    # controlnet_config.in_channels = 320
    controlnet = CustomEnvMapControlNet.from_unet(unet)
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    
    noisy_latents = torch.randn([4,4,64,64])
    timesteps = torch.randint(0, 1000, (4,))
    timesteps = timesteps.long()
    encoder_hidden_states = torch.randn([4,77,1024])
    env_map = torch.rand(4,3,256,512)
    light_enc = EnvMapEncoder(map_size=64,latent_size=320)
    light_feat = light_enc(env_map)
    # controlnet_embedding = ControlNetConditioningEmbedding(conditioning_embedding_channels=320)
    # latent = controlnet_embedding(env_map)
    print(light_feat.shape)
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=env_map,  # 64-channel input
        return_dict=False,
    )
    # for i in range(len(down_block_res_samples)):
    #     print(down_block_res_samples[i].shape)
    # pipe = StableDiffusionControlNetImg2ImgPipeline(
    # vae=vae,
    # text_encoder=None,  # Not using text prompts
    # tokenizer=None,
    # unet=unet,
    # controlnet=controlnet,
    # safety_checker=None,  # Disable safety checker if not needed
    # feature_extractor=None,
    # )
    # with torch.no_grad():
    #     output = pipe(
    #         prompt="",  # Empty string since we're not using text conditioning
    #         image=image,
    #         control_image=env_map,
    #         num_inference_steps=50,
    #         guidance_scale=7.5,
    #     )
    # relit_image = output.images[0]
    # print(relit_image.shape)




    