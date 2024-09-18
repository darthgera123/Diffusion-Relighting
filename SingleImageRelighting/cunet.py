import torch
import torch.nn as nn
from torchvision import models
import timm
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetConditioningEmbedding  # Import the conditioning embedding
from transformers import AutoTokenizer, PretrainedConfig
from argparse import ArgumentParser

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Embedding(num_classes, in_channel*2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out

class AdaIN(nn.Module):
    # def __init__(self, in_channel, num_classes, eps=1e-5):
    #     super().__init__()
    #     self.num_classes = num_classes
    #     self.eps = eps
    #     self.l1 = nn.Linear(num_classes, in_channel*4, bias=True) #bias is good :)
    #     self.emb = nn.Embedding(num_classes, num_classes)
    def __init__(self, in_channel, latent_shape, eps=1e-5):
        super().__init__()
        self.latent_shape = latent_shape
        self.eps = eps
        self.flatten_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]
        
        self.l1 = nn.Linear(self.flatten_dim, in_channel*4, bias=True) #bias is good :)
        

    def c_norm(self, x, bs, ch, eps=1e-7):
        # assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0)==y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        # y_ = self.l1(y).view(bs, ch, -1)
        y_ = y.view(bs, -1)
        y_ = self.l1(y_).view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out =   ((x - x_mean.expand(size)) / x_std.expand(size)) \
                * y_std.expand(size) + y_mean.expand(size)
        return out

class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def b_norm(self, x, bs, eps=1e-5):
        # assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, 1, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, 1, 1, 1)
        return x_std, x_mean

    def forward(self, x):
        size = x.size()
        bs = size[0]
        x_ = x.view(bs, -1)
        x_std, x_mean = self.b_norm(x_, bs)
        out = (x - x_mean.expand(size)) / x_std.expand(size)
        return out

class MakeOneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        ind = torch.argmax(x)
        return nn.functional.one_hot(ind, self.num_classes)

class HalfDropout(nn.Module):
    def __init__(self, p=0.3):
        super(HalfDropout, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        ch = x.size(1)
        a = x[:, :ch//2]
        a = self.dropout(a)
        b = x[:, ch//2:]
        out = torch.cat([a,b], dim=1)
        return out

def upsample_box(out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.BatchNorm2d(out_channels, affine=False)
    )

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2),
        nn.BatchNorm2d(out_channels, affine=False),
        nn.LeakyReLU(0.2, inplace=True)
    )

def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    
def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )   

class LinearAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(LinearAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Linearize the attention to (B, heads, dim_head, H * W)
        qkv = self.to_qkv(x).reshape(B, self.num_heads, 3 * self.head_dim, H * W)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # Compute linear attention
        k_softmax = k.softmax(dim=-1)  # Softmax over key dimensions
        v_weighted = v / (H * W)       # Normalize value by spatial dimensions
        context = torch.einsum('bhnd,bhne->bhde', k_softmax, v_weighted)  # Linear attention computation

        # Use context to compute output
        out = torch.einsum('bhde,bhne->bhnd', context, q)
        out = out.reshape(B, C, H, W)  # Reshape back to original shape

        # Output projection
        return self.to_out(out)
    
class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self,in_ch=6,out_ch=48,latent_size=(4,16,16),attn=False):
        super(Conditional_UNet, self).__init__()

        
        self.dconv_down1 = r_double_conv(in_ch, 64)
        self.dconv_down2 = r_double_conv(64, 128)
        self.dconv_down3 = r_double_conv(128, 256)
        self.dconv_down4 = r_double_conv(256, 512)
        self.attn = attn
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        #self.dropout_half = HalfDropout(p=0.3)
        if self.attn:
            self.mid_block = nn.Sequential(
                r_double_conv(512, 512),
                LinearAttentionBlock(512),  # Linear attention block
                AdaIN(512, latent_shape=latent_size),  # AdaIN for conditioning
                r_double_conv(512, 512)
            )
        
        self.adain3 = AdaIN(512, latent_shape=latent_size)
        self.adain2 = AdaIN(256, latent_shape=latent_size)
        self.adain1 = AdaIN(128, latent_shape=latent_size)

        self.dconv_up3 = r_double_conv(256 + 512, 256)
        self.dconv_up2 = r_double_conv(128 + 256, 128)
        self.dconv_up1 = r_double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, out_ch, 1)
        self.activation = nn.Tanh()
        #self.init_weight() 
        
    def forward(self, x, c):

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)

        if self.attn:

        # Mid-block with linear attention and AdaIN conditioning
            x = self.mid_block[0](x)  # First double convolution
            x = self.mid_block[1](x)  # Linear attention
            x = self.mid_block[2](x, c)  # AdaIN for conditioning
            x = self.mid_block[3](x)  # Final double convolution

        #dropout
        #x = self.dropout_half(x)
        
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.adain2(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)

        x = self.adain1(x, c)
        x = self.upsample(x)        
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class EnvMapEncoder(nn.Module):
    def __init__(self,map_size=16,latent_size=4):
        super(EnvMapEncoder, self).__init__()
        # Load pretrained VGG19 model
        # vgg19 = models.vgg19(pretrained=True)
        efficientnet_b1 = models.efficientnet_b1(pretrained=True)
        # Remove classifier layers and keep only the feature extractor part
        # self.features = vgg19.features
        self.features = nn.Sequential(*list(efficientnet_b1.children())[:-2])
        
        # Define a custom additional layers to adjust the output size to 4x16x16
        # self.additional_layers = nn.Sequential(
        #     nn.Conv2d(512, latent_size, kernel_size=1),  # Reduces channel size to 4
        #     nn.AdaptiveMaxPool2d((map_size, map_size))     # Resize feature maps to 16x16
        # )
        self.additional_layers = nn.Sequential(
            nn.Conv2d(1280, latent_size, kernel_size=1),  # Reduces channel size to 4, note channel count change
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((map_size, map_size))    # Resize feature maps to 16x16
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)

        # Transform feature maps to the desired output size
        x = self.additional_layers(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,map_size=16,latent_size=4,pretrained=True):

        super(TransformerEncoder,self).__init__()
        # self.swin_transformer = timm.create_model('swin_base_patch4_window7_224_in22k', 
        #                                           pretrained=pretrained, 
        #                                           features_only=True, 
        #                                           out_indices=(3,),
        #                                           img_size=(256,512))
        self.mobilevit = timm.create_model('mobilevit_s', 
                                                 pretrained=pretrained, 
                                                 features_only=True, 
                                                 out_indices=(3,),
                                                 img_size=(256,512))
        self.latent_size = latent_size
        self.map_size = map_size

        # self.additional_layers = nn.Sequential(
        #     nn.Conv2d(8,latent_size,kernel_size=1),
        #     nn.AdaptiveAvgPool2d((map_size,map_size))
        # )
        self.additional_layers = nn.Sequential(
            nn.Conv2d(128,latent_size,kernel_size=1),
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((map_size,map_size))
        )
    def forward(self,x):
        x = self.mobilevit(x)[0]
        x = self.additional_layers(x)
        return x



class CondUnet(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,latent_size=(4,16,16),light_encoder='cnn',attn=False):
        super(CondUnet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.unet = Conditional_UNet(in_ch=in_ch,out_ch=out_ch,latent_size=latent_size,attn=attn)
        if light_encoder == 'cnn':
            self.envmap_encoder = EnvMapEncoder(map_size=16,latent_size=4)
        elif light_encoder == 'transformer':
            self.envmap_encoder = TransformerEncoder(map_size=16,latent_size=4)

    def forward(self,image,envmap):
        
        latent = self.envmap_encoder(envmap)
        
        relit = self.unet(image,latent)
        return relit
    


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1000:
        return f"{total_params} parameters"
    elif total_params < 1_000_000:
        return f"{total_params / 1_000:.1f}K parameters"  # Kilos
    else:
        return f"{total_params / 1_000_000:.1f}M parameters"  # Millions

class CustomControlNet(ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels = self.block_out_channels[0],
            block_out_channels=self.conditioning_embedding_out_channels,
            conditioning_channels=320,
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

if __name__ == "__main__":
    
    args = parse_args()

    num_classes = 6
    batch_size = 4
    # cunet = Conditional_UNet(num_classes=num_classes,in_ch=6,out_ch=48)
    
    envmap = EnvMapEncoder(map_size=64,latent_size=320)
    # envmap = TransformerEncoder(map_size=320,latent_size=64)
    light = torch.rand(4,3,256,512)

    env_map_feature  = envmap(light)
    print("Env map",env_map_feature.shape)
    
    image = torch.rand(4,3,512,512)
    
    # print(one_hot_tensor.shape)
    # classes = torch.rand(4,6)
    # feature = torch.rand(batch_size,512,32,32)
    # print(latent.shape)
    latent_size = (4,16,16)
    # adain = AdaIN(in_channel=512, latent_shape=latent_size)
    # output = adain(feature,latent)
    # print(output.shape)
    # cunet = Conditional_UNet(in_ch=6,out_ch=48,latent_size=latent_size)
    # cunet = CondUnet(in_ch=3,out_ch=3,latent_size=latent_size,light_encoder='cnn',attn=False)
    # output = cunet(image,light)
    # print(count_parameters(envmap))
    # print(latent.shape)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/'
    )
    controlnet = ControlNetModel.from_unet(unet)
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",cache_dir='/scratch/inf0/user/pgera/FlashingLights/SingleRelight/cache/')
    
    # custom_controlnet = CustomControlNet.from_unet(unet)
    noisy_latents = torch.randn([4,4,64,64])
    timesteps = torch.randint(0, 1000, (4,))
    timesteps = timesteps.long()
    encoder_hidden_states = torch.randn([4,77,1024])
    env_map = torch.rand(4,1,512,512)
    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=env_map,  # 64-channel input
        return_dict=False,
    )
    for i in range(len(down_block_res_samples)):
        print(down_block_res_samples[i].shape)
    # print(down_block_res_samples[0].shape)
    print(mid_block_res_sample.shape)