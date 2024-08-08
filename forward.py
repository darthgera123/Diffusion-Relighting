"""
Forward diffusion model
We implement linear noise schedule to apply loss
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from data import DiffusionDataset,custom_collate
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from imageio.v2 import imread,imwrite


def linear_beta_scheduler(timesteps,start=1e-4,end=2e-2):
    """
    linearly increase the beta variable
    """
    return torch.linspace(start,end,timesteps)

def cosine_beta_scheduler(timesteps,s=8e-3):
    """
    beta is a cosine function. destroys information slower
    """
    steps = timesteps+1
    x = torch.linspace(0,timesteps,steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas,1e-4,0.9999)

def quadratic_beta_scheduler(timesteps,start=1e-4,end=2e-2):
    """
    quadraticly increase the beta variable
    """
    return torch.linspace(start**0.5,end**0.5,timesteps)**2

def get_index(vals,t,x_shape):
    """
    Returns specific index t of a passed list of values
    along batch dimension.
    """
    b_size = t.shape[0]
    out = vals.gather(-1,t)
    return out.reshape(b_size, *((1,) * (len(x_shape)-1))).to(t.device) 

def compute_alphas(betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0) # [T] starts from 0.9999 * 0.9998
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1],(1,0),value=1.0) # [T] starts from 1
    sqrt_recip_alphas = torch.sqrt(1.0/alphas)
    # q(xt | x{t-1}) 
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    # q(x{t-1}|xt,x0)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) #precomputing all these values

    return sqrt_recip_alphas,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod,posterior_variance


def forward_process(x0, t,betas,device='cpu'):
    """
    Given x0 and timestep and returns xT
    """
    
    epsilon = torch.randn_like(x0)
    sqrt_recip_alphas,\
        sqrt_alphas_cumprod,\
        sqrt_one_minus_alphas_cumprod,\
        posterior_variance = compute_alphas(betas)
    
    sqrt_alphas_cumprod_t = get_index(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index(sqrt_one_minus_alphas_cumprod, t, x0.shape)

    #mean + variance
    return sqrt_alphas_cumprod_t.to(device)* x0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device)* epsilon.to(device), epsilon.to(device)

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--images",default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    
    args = parse_args()

    IMG_SIZE = 128
    BATCH_SIZE = 32
    img_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t*2)-1) #Scale data between -1,1
    ])

    diff_dataset = DiffusionDataset(image_path=args.images,transforms=img_transforms)

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1,2,0)), #change channel dimension
        transforms.Lambda(lambda t: (t*255).numpy().astype('uint8'))
    ])

    img_dataloader = DataLoader(diff_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1,collate_fn=custom_collate)
    
    x_start = next(iter(img_dataloader))[0]
    np_x0 = reverse_transforms(x_start)
    
    
    
    
    T = 300
    betas = cosine_beta_scheduler(timesteps=T) 
    noisy_images = [np_x0]
    for t in range(0,T,50):
        timestep = torch.tensor([t])
        x_t,epsilon = forward_process(x_start,timestep,betas)
        noisy_images.append(reverse_transforms(x_t))
    
    np_img = np.hstack(noisy_images)
    imwrite('cosine_sample.png',np_img)

    betas = linear_beta_scheduler(timesteps=T) 
    noisy_images = [np_x0]
    for t in range(0,T,50):
        timestep = torch.tensor([t])
        x_t,epsilon = forward_process(x_start,timestep,betas)
        noisy_images.append(reverse_transforms(x_t))
    
    np_img = np.hstack(noisy_images)
    imwrite('linear_sample.png',np_img)

    betas = quadratic_beta_scheduler(timesteps=T) 
    noisy_images = [np_x0]
    for t in range(0,T,50):
        timestep = torch.tensor([t])
        x_t,epsilon = forward_process(x_start,timestep,betas)
        noisy_images.append(reverse_transforms(x_t))
    
    np_img = np.hstack(noisy_images)
    imwrite('quadratic_sample.png',np_img)





