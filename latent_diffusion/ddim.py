import torch
from torch import nn
from scheduler import get_beta_schedule,compute_alphas,get_index


class DDIM_scheduler(nn.Module):
    def __init__(self,
                 num_timesteps=100,
                 train_timesteps = 1000,
                 clip_sample=True,
                 noise_schedule='linear') -> None:
        super().__init__()
        """
        DDPM destroys image step by step and is markovian
        DDIM approximates the effect of many noise steps and destroys by taking large
        deterministic steps. It is non markovian
        We can almost achieve the same effect as DDPM in 50 steps instead of 1000
        Same forward process, lesser steps in backward process
        """

        self.num_timesteps = num_timesteps
        self.train_timesteps = train_timesteps
        self.ratio = self.train_timesteps//self.num_timesteps
        self.scheduler = noise_schedule
        self.final_alpha_cumprod = torch.tensor([1.0])
        self.clip_sample = clip_sample
        
        # Define it as register_buffer as these are non-learnable parameters
        # They would be included in the model state for loading but not learnt
        # .requires_grad=False is an alternative but its messy and not loading or unloading

        betas = get_beta_schedule(self.scheduler,self.timesteps)
        sqrt_recip_alphas,\
            sqrt_alphas_cumprod,\
            sqrt_one_minus_alphas_cumprod,\
            posterior_variance = compute_alphas(betas)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",1-betas)
        self.register_buffer("alphas_cumprod",torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_recip_alphas",sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod",sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance",posterior_variance)

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self,x_t,t,z_t,eta=0):
        """
        Given approximation of noise z_t in x_t
        predict x_(t-1)
        """
        b,c,h,w = z_t.shape
        device = z_t.device

        #compute steps based on the ratio
        t = t*self.ratio
        t_prev = t - self.ratio

        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.ge(0),self.final_alpha_cumprod.to(device)) #>=0
        alpha_cumprod_prev_t = alpha_cumprod_prev.view(b,1,1,1)
        alpha_cumprod_prev_sqrt = self.sqrt_alphas_cumprod[t_prev]

        #estimate origin
        x_0_pred = self.estimate_origin(x_t,t,z_t)
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred,-1,1)
        
        std_dev_t = eta * self.estimate_std(t,t_prev).view(b,1,1,1)
        x_0_grad = (1-alpha_cumprod_prev_t - std_dev_t**2).sqrt() * z_t
        prev_sample = alpha_cumprod_prev_sqrt * x_0_pred + x_0_grad

        if eta > 0:
            noise = torch.randn(model_output.shape,dtype=model_output.dtype)
            prev_sample = prev_sample+std_dev_t*eta*noise
        return prev_sample
    
    
    def estimate_std(self,t,t_prev):
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.gt(0),self.final_alpha_cumprod.to(alpha_cumprod.device))
        one_minus_alpha_cumprod = 1-alpha_cumprod
        one_minus_alpha_cumprod_prev = 1-alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev/one_minus_alpha_cumprod)* (1-alpha_cumprod/alpha_cumprod_prev)
        return var.sqrt()
    
    def estimate_origin(self,x_t,t,z_t):
        alpha_cumprod = self.alphas_cumprod.view(b,1,1,1)
        alpha_one_minus_cumprod_sqrt = alphas_one_minus_cumprod_sqrt[t]
        return (x_t-alpha_one_minus_cumprod_sqrt*z_t)/alpha_cumprod.sqrt()



if __name__ == "__main__":
    batch = 4
    img_size = 32
    timesteps = 100
    ddim = DDIM_scheduler()
    
    
    x_t = torch.rand(batch,3,img_size,img_size)
    noise = torch.randn_like(x_t)
    t = torch.randint(0, timesteps, (batch,), ).long()
    
    noise_pred = ddim(x_t,t,noise)

    print(noise_pred.shape)
    