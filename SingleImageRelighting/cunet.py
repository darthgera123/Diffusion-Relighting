import torch
import torch.nn as nn
from torchvision import models

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


class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self,in_ch=6,out_ch=48,latent_size=(4,16,16)):
        super(Conditional_UNet, self).__init__()

        
        self.dconv_down1 = r_double_conv(in_ch, 64)
        self.dconv_down2 = r_double_conv(64, 128)
        self.dconv_down3 = r_double_conv(128, 256)
        self.dconv_down4 = r_double_conv(256, 512)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        #self.dropout_half = HalfDropout(p=0.3)
        
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
            nn.AdaptiveMaxPool2d((map_size, map_size))    # Resize feature maps to 16x16
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Transform feature maps to the desired output size
        x = self.additional_layers(x)
        return x

class CondUnet(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,latent_size=(4,16,16)):
        super(CondUnet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.unet = Conditional_UNet(in_ch=in_ch,out_ch=out_ch,latent_size=latent_size)
        self.envmap_encoder = EnvMapEncoder(map_size=16,latent_size=4)

    def forward(self,image,envmap):
        
        latent = self.envmap_encoder(envmap)
        
        relit = self.unet(image,latent)
        return relit


if __name__ == "__main__":
    
    num_classes = 6
    batch_size = 4
    # cunet = Conditional_UNet(num_classes=num_classes,in_ch=6,out_ch=48)
    
    envmap = EnvMapEncoder(map_size=16,latent_size=4)
    light = torch.rand(4,3,256,512)
    latent  = envmap(light)
    image = torch.rand(4,3,512,512)
    
    # print(one_hot_tensor.shape)
    classes = torch.rand(4,6)
    feature = torch.rand(batch_size,512,32,32)
    print(latent.shape)
    latent_size = (4,16,16)
    adain = AdaIN(in_channel=512, latent_shape=latent_size)
    output = adain(feature,latent)
    print(output.shape)
    # cunet = Conditional_UNet(in_ch=6,out_ch=48,latent_size=latent_size)
    cunet = CondUnet(in_ch=3,out_ch=3,latent_size=latent_size)
    output = cunet(image,light)
    print(output.shape)
    