import torch
from torch import nn
from scheduler import get_beta_schedule,compute_alphas,get_index


class DDPM_scheduler(nn.Module):
    def __init__(self,
                 timesteps=1000,
                 noise_schedule='linear') -> None:
        super().__init__()

        self.timesteps = timesteps
        self.scheduler = noise_schedule
        
        # Define it as register_buffer as these are non-learnable parameters
        # They would be included in the model state for loading but not learnt
        # .requires_grad=False is an alternative but its messy and not loading or unloading

        betas = get_beta_schedule(self.scheduler,self.timesteps)
        sqrt_recip_alphas,\
            sqrt_alphas_cumprod,\
            sqrt_one_minus_alphas_cumprod,\
            posterior_variance = compute_alphas(betas)
        self.register_buffer("betas",betas)
        self.register_buffer("sqrt_recip_alphas",sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod",sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance",posterior_variance)

    def forward(self,*args,**kwargs):
        return self.step(*args,**kwargs)
    
    def step(self,x_t,t,z_t):
        """
        Given approximation of noise z_t in x_t
        predict x_(t-1)
        """
        # Approx Distribution of previous sample in chain
        mean_pred, std_pred = self.posterior_params(x_t,t,z_t)

        # sample from distribution
        z = torch.randn_like(x_t) if any(t>0) else torch.zeros_like(x_t)
        return mean_pred + z*std_pred
    
    def posterior_params(self,x_t,t,noise_pred):

        sqrt_alphas_cumprod_t = get_index(self.sqrt_alphas_cumprod,t,x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index(self.sqrt_one_minus_alphas_cumprod,t,x_t.shape)
        sqrt_recip_alphas_t = get_index(self.sqrt_recip_alphas,t,x_t.shape)
        beta_t = get_index(self.betas,t,x_t.shape)
        mean = sqrt_recip_alphas_t * (x_t - beta_t* noise_pred/ sqrt_one_minus_alphas_cumprod_t)
        std = beta_t

        # std = sqrt_one_minus_alphas_cumprod_t
        

        return mean,std

if __name__ == "__main__":
    batch = 4
    img_size = 32
    timesteps = 100
    ddpm = DDPM_scheduler()
    
    
    x_t = torch.rand(batch,3,img_size,img_size)
    noise = torch.randn_like(x_t)
    t = torch.randint(0, timesteps, (batch,), ).long()
    
    noise_pred = ddpm(x_t,t,noise)

    print(noise_pred.shape)
    