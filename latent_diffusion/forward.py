import torch
import torch.nn as nn
from scheduler import get_beta_schedule, compute_alphas

class GaussianForwardProcess(nn.Module):

    def __init__(self,
                 num_timesteps=1000,
                 noise_schedule = 'linear'
                 ) -> None:
        super().__init__()
        self.schedule = noise_schedule
        self.num_timesteps = num_timesteps

        betas = get_beta_schedule(self.schedule,self.num_timesteps)
        sqrt_recip_alphas,\
            sqrt_alphas_cumprod,\
            sqrt_one_minus_alphas_cumprod,\
            posterior_variance = compute_alphas(betas)
        self.register_buffer("betas",betas)
        self.register_buffer("betas_sqrt",torch.sqrt(self.betas))
        self.register_buffer("alphas",1-betas)
        self.register_buffer("sqrt_alphas",torch.sqrt(self.alphas))
        self.register_buffer("alphas_cumprod",torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_recip_alphas",sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod",sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance",posterior_variance)

    @torch.no_grad()
    def forward(self, x_0, t, return_noise=False):
        """
        q(x_t|x_0) = N(x_t,alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        b = x_0.shape[0]
        mean = x_0*self.sqrt_alphas_cumprod[t].view(b,1,1,1)
        std = self.sqrt_one_minus_alphas_cumprod[t].view(b,1,1,1)
        noise = torch.randn_like(x_0)
        output = mean+std*noise

        if not return_noise:
            return output
        else:
            return output,noise
    
    @torch.no_grad()
    def step(self,x_t,t,return_noise=False):
        """
        Sampling
        q(x_t|x_(t-1)) = N(x_t;alphas_sqrt(t)*x_0,betas(t)*I)
        """
        mean = self.alphas_sqrt[t]*x_t
        std = self.betas_sqrt[t]

        noise =torch.randn_like(x_t)
        output=mean+noise*std
        if not return_noise:
            return output
        else:
            return output,noise
