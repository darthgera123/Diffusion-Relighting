import torch
import torch.nn as nn
from model_utils import default,ConvNextBlock,\
    ResnetBlock,PosEmbedding, Attention, LinearAttention, \
    Residual, Upsample, Downsample, PreNorm, exists

from functools import partial

class UNet(nn.Module):
    """
    UNet module consisting of ResNet/ConvNext blocks. 
    Input: Noisy Image and Corresponding Timestep
    Output: Noise
    Time step is passed through an MLP and we get an embedding
    Along with image that is passed via every downsampling and upsampling block
    The Residual blocks also consist of linear attention blocks which are very important
    This Network can be made even more complex
    """
    def __init__(self,dim,init_dim=None,out_dim=None,
                 dim_mults=(1,2,4,8),channels=3,time_emb=True,resnet_block_groups=8,
                 use_convnext=False,convnext_mult=2) -> None:
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                PosEmbedding(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self,x,time):
        x = self.init_conv(x)
        
        if self.time_mlp:
            t = self.time_mlp(time)
        else:
            t = None
        h=[]
        # downsample
        for block1,block2,attn,downsample in self.downs:
            x = block1(x,t)
            x = block2(x,t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        # bottleneck
        x = self.mid_block1(x,t)
        x = self.mid_attn(x)
        x = self.mid_block2(x,t)
        # upsample
        for block1,block2,attn,upsample in self.ups:
            x = torch.cat((x,h.pop()),dim=1)
            x = block1(x,t)
            x = block2(x,t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1000:
        return f"{total_params} parameters"
    elif total_params < 1_000_000:
        return f"{total_params / 1_000:.1f}K parameters"  # Kilos
    else:
        return f"{total_params / 1_000_000:.1f}M parameters"  # Millions

if __name__ == "__main__":
    batch = 4
    img_size = 32
    timesteps = 100
    
    
    model = UNet(
        dim=img_size,
        channels=3,
        dim_mults=(1,2,4,8)
    )
    
    x_t = torch.rand(batch,3,img_size,img_size)
    t = torch.randint(0, timesteps, (batch,), ).long()
    pred_noise = model(x_t,t)
    print(pred_noise.shape)
    print(count_parameters(model))










