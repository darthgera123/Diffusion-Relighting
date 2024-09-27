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
from dataset import PortraitControlNetDataset
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline

)
from transformers import AutoTokenizer, PretrainedConfig


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
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
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
            self.controlnet_cond_embedding.additional_layers
        )

        
def collate_fn_relit(examples):
    pixel_values = torch.stack([example["diffuse"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["env"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    relit_pixel_values = torch.stack([example["relit"] for example in examples])
    relit_pixel_values = relit_pixel_values.to(memory_format=torch.contiguous_format).float()

    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    attention_mask = attention_mask.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["text"] for example in examples])
    input_ids = input_ids.to(memory_format=torch.contiguous_format)

    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask": mask,
        "relit": relit_pixel_values
    }




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
def save_tensor_as_image(tensor_image, file_name):
    # If the tensor has a batch dimension, remove it
    if tensor_image.ndim == 4:
        tensor_image = tensor_image.squeeze(0)  # Removes the batch dimension

    # Ensure the tensor is in the range [0, 1] (for normalization)
    tensor_image = torch.clamp(tensor_image, 0, 1)

    # Convert tensor from (C, H, W) -> (H, W, C) and multiply by 255 to get pixel values
    image_np = tensor_image.permute(1, 2, 0).cpu().numpy() * 255

    # Convert to uint8
    image_np = image_np.astype('uint8')

    # Convert numpy array to PIL image
    image_pil = Image.fromarray(image_np)

    # Save the image
    image_pil.save(file_name)



if __name__ == "__main__":
    
    args = parse_args()

    num_classes = 6
    batch_size = 4
    light = torch.rand(4,3,256,512)

    
    image = torch.rand(4,3,512,512)
    tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
        )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, \
        subfolder="unet", revision=args.revision, variant=args.variant,\
        cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, \
                                        subfolder="vae",\
                                        cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    
    weight_dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # controlnet_config.in_channels = 320
    controlnet = CustomEnvMapControlNet.from_unet(unet).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler",
                                                    cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    
    val_dataset = PortraitControlNetDataset(relit_path=args.relit_path, env_path=args.env_path, \
                              diffuse_path=args.diffuse_path, mask_path=args.mask_path,\
                              caption='Reconstruction',tokenizer=tokenizer,\
                              normal_path=args.normal_path,train=False,crop=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=collate_fn_relit,
        batch_size=1,
        num_workers=1,
    )
    
    # pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=unet,
    #     controlnet=controlnet,
    #     safety_checker=None,
    #     revision=args.revision,
    #     variant=args.variant,
    #     torch_dtype=weight_dtype,
    # )
    pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,  # Optional, can be enabled for safety checks
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    # pipeline.eval()

    noisy_latents = torch.randn([4,4,64,64]).to(device)
    timesteps = torch.randint(0, 1000, (4,)).to(device)
    timesteps = timesteps.long()
    encoder_hidden_states = torch.randn([4,77,1024]).to(device)
    env_map = torch.rand(4,3,256,512).to(device)
    light_enc = EnvMapEncoder(map_size=64,latent_size=320).to(device)
    light_feat = light_enc(env_map)
    # controlnet_embedding = ControlNetConditioningEmbedding(conditioning_embedding_channels=320)
    # latent = controlnet_embedding(env_map)
    
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=env_map,  # 64-channel input
        return_dict=False,
        
    )
    with torch.no_grad():
        for step,batch in enumerate(val_dataloader):
            albedo = batch['pixel_values'].to(device,dtype=weight_dtype)
            text = batch['input_ids'].to(device,dtype=weight_dtype)
            env_map = batch['conditioning_pixel_values'].to(device,dtype=weight_dtype)
            relit = batch['relit'].to(device,dtype=weight_dtype)
            
            prompts = ['Reconstruction']*1
            print(albedo.shape,len(prompts),env_map.shape)
            image = pipeline(
                prompt= prompts, control_image = env_map,image=albedo, num_inference_steps=20
            ).images
            print(len(image))
            image[0].save('output.png')
            save_tensor_as_image(albedo[0],'albedo.png')
            save_tensor_as_image(env_map[0],'envmap.png')
            save_tensor_as_image(relit[0],'relit.png')
            exit()




    