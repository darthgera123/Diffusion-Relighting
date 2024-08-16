"""
Training code for LatentDiffusion Model
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.models import AutoencoderKL

from denoise_diffusion import DenoisingDiffusion, ConditionalDenoisingDiffusion

class AutoEncoderDiffusion(nn.Module):
    """
    Wrapper for pretrained AutoEncoder module
    We use a frozen Autoencoder, obtain latents and run DDPM on latent space
    """
    def __init__(self,
                 model_type=['stabilityai/sd-vae-ft-ema']) -> None:
        super().__init__()
    
        self.model = AutoencoderKL.from_pretrained(model_type)

    def forward(self,input):
        return self.model(input).sample
    
    def encode(self,input,mode=False):
        dist = self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample
    

class LatentDiffusion(pl.LightningModule):
    
    def __init__(self,
                 train_dataset,
                 val_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        
        super.__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = val_dataset

        self.lr = lr
        self.register_buffer('latent_scale_factor',torch.tensor(latent_scale_factor))

        self.batch_size = batch_size
        self.autoencoder = AutoEncoderDiffusion()
        with torch.no_grad():
            self.latent_dim = self.autoencoder.encode(
                torch.ones(1,3,256,256)
            ).shape[1]
        self.model = DenoisingDiffusion(channels=self.latent_dim,
                                        num_timesteps=num_timesteps)
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):
        return self.output_T(self.autoencoder.decode(self.model(*args,**kwargs)/self.latent_scale_factor))
        
    def input_T(self,input):
        # Convert to [-1,1]
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self,input):
        # Convert to [0,1]
        return (input.add_(1)).div_(2)

    def training_step(self,batch,batch_idx):
        """
        Encode the input images to a latent space
        Then pass it to diffusion model
        """
        latents = self.autoencoder.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        self.log('train_loss',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        latents = self.autoencoder.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        self.log('val_loss',loss)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4)
        else:
            return None
    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p : p.requires_grad,self.model_parameters())),lr=self.lr)
    
class ConditionalLatentDiffusion(pl.LightningModule):
    
    def __init__(self,
                 train_dataset,
                 val_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        
        super.__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = val_dataset

        self.lr = lr
        self.register_buffer('latent_scale_factor',torch.tensor(latent_scale_factor))

        self.batch_size = batch_size
        self.autoencoder = AutoEncoderDiffusion()
        with torch.no_grad():
            self.latent_dim = self.autoencoder.encode(
                torch.ones(1,3,256,256)
            ).shape[1]
        self.model = ConditionalDenoisingDiffusion(channels=self.latent_dim,
                                        cond_channels=self.latent_dim,
                                        num_timesteps=num_timesteps)
        
    @torch.no_grad()
    def forward(self,condition, *args,**kwargs):
        condition_latent = self.autoencoder.encode(self.input_T(condition.to(self.device))).detach() * self.latent_scale_factor
        output_code = self.model(condition_latent,*args,**kwargs)/self.latent_scale_factor

        return self.output_T(self.autoencoder.decode(output_code))
        
    def input_T(self,input):
        # Convert to [-1,1]
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self,input):
        # Convert to [0,1]
        return (input.add_(1)).div_(2)

    def training_step(self,batch,batch_idx):
        """
        Encode the input images to a latent space
        Then pass it to diffusion model
        """
        condition,output = batch
        with torch.no_grad():
            latents = self.autoencoder.encode(self.input_T(output)).detach() * self.latent_scale_factor
            latents_condition = self.autoencoder.encode(self.input_T(condition)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)
        self.log('train_loss',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        condition,output = batch
        with torch.no_grad():
            latents = self.autoencoder.encode(self.input_T(output)).detach() * self.latent_scale_factor
            latents_condition = self.autoencoder.encode(self.input_T(condition)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)
        self.log('val_loss',loss)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4)
        else:
            return None
    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p : p.requires_grad,self.model_parameters())),lr=self.lr)