import torch
import torch.nn.functional as F

def get_beta_schedule(variant, timesteps):
    
    if variant=='cosine':
        return cosine_beta_scheduler(timesteps)
    elif variant=='linear':
        return linear_beta_scheduler(timesteps)
    elif variant=='quadratic':
        return quadratic_beta_scheduler(timesteps)
    elif variant=='sigmoid':
        return sigmoid_beta_scheduler(timesteps)
    else:
        raise NotImplemented


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

def sigmoid_beta_scheduler(timesteps,start=1e-4,end=2e-2):
    """
    beta is a sigmoid function. destroys information slower
    """
    betas = torch.linspace(-6,6,timesteps)
    return torch.sigmoid(betas) * (end-start) + start

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