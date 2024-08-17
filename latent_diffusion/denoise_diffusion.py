import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm
from unet_backbone import UnetConvNextBlock
from ddpm import DDPM_scheduler
from forward import GaussianForwardProcess

class DenoisingDiffusion(nn.Module):

    def __init__(self,
                 channels=3,
                 loss_type='l2',
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None) -> None:
        super().__init__()

        self.channels = channels
        self.num_timesteps = num_timesteps

        if loss_type == 'l1':
            self.criterion = F.l1_loss
        elif loss_type == 'l2':
            self.criterion = F.mse_loss
        elif loss_type == 'huber':
            self.criterion = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        
        self.forward_process = GaussianForwardProcess(num_timesteps=num_timesteps,
                                                      noise_schedule=schedule)
        
        self.model = UnetConvNextBlock(dim=64,
                                       dim_mults=(1,2,4,8),
                                       channels=self.channels,
                                       out_dim=self.channels,
                                       with_time_emb=True)
        self.sampler = DDPM_scheduler(timesteps=self.num_timesteps) if sampler is None else sampler

    @torch.no_grad()
    def forward(self,
                shape=(256,256),
                batch_size=1,
                sampler=None,
                verbose=False):
    
        b,c,h,w = batch_size, self.channels,*shape
        device = next(self.model.parameters()).device

        if sampler is None:
            sampler = self.sampler
        else:
            sampler = sampler.to(device)

        num_timesteps = sampler.num_timesteps
        it = reversed(range(0,num_timesteps))

        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t = self.model(x_t,t) # prediction of noise
            x_t = sampler(x_t,t,z_t) #prediction of next state
        
        return x_t
    
    def p_loss(self,output):
        """
        Output: [-1,1]
        """
        b,c,h,w = output.shape
        device = output.device

        t = torch.randint(0,self.forward_process.num_timesteps,(b,),device=device).long()
        output_noisy, noise = self.forward_process(output,t,return_noise=True)

        noise_pred = self.model(output_noisy,t)

        return self.criterion(noise,noise_pred)
    
class ConditionalDenoisingDiffusion(nn.Module):

    def __init__(self,
                 channels=3,
                 cond_channels=3,
                 loss_type='l2',
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None) -> None:
        super().__init__()

        self.channels = channels
        self.cond_channels = cond_channels
        self.num_timesteps = num_timesteps

        if loss_type == 'l1':
            self.criterion = F.l1_loss
        elif loss_type == 'l2':
            self.criterion = F.mse_loss
        elif loss_type == 'huber':
            self.criterion = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        
        self.forward_process = GaussianForwardProcess(num_timsteps=num_timesteps,
                                                      schedule=schedule)
        
        self.model = UnetConvNextBlock(dim=64,
                                       dim_mults=(1,2,4,8),
                                       channels=self.channels,
                                       out_dim=self.channels,
                                       with_time_emb=True)
        self.sampler = DDPM_scheduler(num_timsteps=self.num_timesteps) if sampler is None else sampler

    @torch.no_grad()
    def forward(self,
                condition,
                shape=(256,256),
                batch_size=1,
                sampler=None,
                verbose=False):
    
        b,c,h,w = batch_size, self.channels,*shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        if sampler is None:
            sampler = self.sampler
        else:
            sampler = sampler.to(device)

        num_timesteps = sampler.num_timesteps
        it = reversed(range(0,num_timesteps))

        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_input = torch.cat([x_t,condition],1).to(device)
            z_t = self.model(model_input,t) # prediction of noise
            x_t = sampler(x_t,t,z_t) #prediction of next state
        
        return x_t
    
    def p_loss(self,output,condition):
        """
        Output: [-1,1]
        """
        b,c,h,w = output.shape
        device = output.device

        t = torch.randint(0,self.forward_process.num_timesteps,(b,),device=device).long()
        output_noisy, noise = self.forward_process(output,t,return_noise=True)

        model_input = torch.cat([output_noisy,condition],1).to(device)
        noise_pred = self.model(output_noisy,t)

        return self.criterion(noise,noise_pred)

        