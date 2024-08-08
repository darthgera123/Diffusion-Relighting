import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from data import DiffusionDataset,custom_collate
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.utils import save_image
import os
from imageio.v2 import imread,imwrite
from forward import linear_beta_scheduler,\
    cosine_beta_scheduler,compute_alphas,get_index
from model import UNet
from data import DiffusionDataset
# Define a diffusion class and make them all part of this


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class DDPM:
    """
    3 Parts to this class
    Noise scheduler
    Forward diffusion
    Model to denoise
    Training step
    Sampling step
    """
    def __init__(self,
                img_dataloader,
                batch_size,
                img_size,
                loss_type='huber',
                noise='linear',
                timesteps=300, 
                log_dir='runs/exp',
                save_dir='val',
                lr=1e-3,
                channels=3,
                device='cuda'):

        self.img_dataloader = img_dataloader
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.img_size = img_size
        self.channels = channels
        self.device = device

        if noise == 'linear':
            self.noise_sched = linear_beta_scheduler
        elif noise == 'cosine':
            self.noise_sched = cosine_beta_scheduler
        else:
            raise NotImplementedError
        
        self.betas = self.noise_sched(timesteps=self.timesteps).to(self.device)
        
        self.sqrt_recip_alphas,\
            self.sqrt_alphas_cumprod,\
            self.sqrt_one_minus_alphas_cumprod,\
            self.posterior_variance = compute_alphas(self.betas)
        
            
        self.model =UNet(
                        dim=self.img_size,
                        channels=self.channels,
                        dim_mults=(1,2,4,8)
                    )
        self.model = self.model.to(self.device)  
        self.optim = optim.Adam(self.model.parameters(),lr=lr)      

        if loss_type == 'l1':
            self.criterion = F.l1_loss
        elif loss_type == 'l2':
            self.criterion = F.mse_loss
        elif loss_type == 'huber':
            self.criterion = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        os.makedirs(save_dir,exist_ok=True)

    def forward_process(self, x0, t,epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(x0).to(self.device)
        sqrt_alphas_cumprod_t = get_index(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        return sqrt_alphas_cumprod_t*x0 + sqrt_one_minus_alphas_cumprod_t * epsilon 
    
    # Sampling code
    @torch.no_grad()
    def p_sample(self,x,t,t_index):
        betas_t = get_index(self.betas,t,x.shape)
        sqrt_alphas_cumprod_t = get_index(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index(self.sqrt_recip_alphas, t,x.shape)

        x_prev = self.model(x,t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t*x_prev/sqrt_one_minus_alphas_cumprod_t
        )
        if t_index > 0:
            posterior_variance_t = get_index(posterior_variance, t, x.shape)
            epsilon = torch.randn_like(x).to(self.device)
            return model_mean + torch.sqrt(posterior_variance_t)*epsilon
        else:
            return model_mean
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        b = shape[0]
        img = torch.randn(shape).to(self.device)
        imgs = []
        for i in tqdm(reversed(range(0,timesteps)), desc='sampling loop timestep',total=timesteps):
            img = self.p_sample(img,torch.full((b,),i,device=self.device,dtype=torch.long),i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return self.p_sample_loop(shape=(batch_size,self.channels,self.img_size,self.img_size))

    
    def train_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.img_dataloader),total=len(self.img_dataloader),
                        desc=f'Epoch:{epoch+1}',leave=False)
        
        for batch_idx,data_idx in progress_bar:
            self.optim.zero_grad()
            data_idx = data_idx.to(self.device)
            t = torch.randint(0,self.timesteps,(self.batch_size,),device=self.device).long()
            noise = torch.randn_like(data_idx)
            x_noisy = self.forward_process(x0=data_idx, t=t)
            predicted_noise = self.model(x_noisy,t)
            
            loss = self.criterion(noise,predicted_noise)

            loss.backward()
            self.optim.step()
            total_loss += loss.item()

            progress_bar.set_postfix({'loss':'{:.6f}'.format(loss.item())})
        
        avg_loss = total_loss/len(self.img_dataloader)
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1} | Train Loss: {avg_loss:.6f}')

    
    def save_prediction(self,epoch):
        batches = num_to_groups(4,self.batch_size)
        all_images_list = list(map(lambda n: self.sample(batch_size=n), batches))
        all_images = torch.cat(all_images_list,dim=0)
        all_images= (all_images+1) * 0.5
        save_image(all_images, str(self.save_dir / f'sample-{epoch}.png' , nrow=6))

    def save_checkpoint(self,epoch):
        path = os.path.join(self.save_dir, f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },path)
    
    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def fit(self,epochs,validate_every_n_epochs=20,save_model=100):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            if (epoch+1) % validate_every_n_epochs == 0:
                self.save_prediction(epoch)
            if (epoch+1) % save_model == 0:
                self.save_checkpoint(epoch)

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--images",type=str,default="")
    parser.add_argument("--save_dir",type=str,default="")
    parser.add_argument("--noise",type=str,default="linear")
    parser.add_argument("--log_dir",type=str,default="")
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--timesteps",type=int,default=300)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()

    IMG_SIZE = 32   
    BATCH_SIZE = 128

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

    img_dataloader = DataLoader(diff_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1,drop_last=True)

    ddpm = DDPM(img_dataloader=img_dataloader,
                batch_size=BATCH_SIZE,
                img_size=IMG_SIZE,
                noise=args.noise,
                timesteps=args.timesteps
                )
    """
    Check Forward diffusion
    x_start = next(iter(img_dataloader))[0]
    np_x0 = reverse_transforms(x_start)
    noisy_images = [np_x0]
    for t in range(0,args.timesteps,50):
        timestep = torch.tensor([t]).to('cuda')
        x_t = ddpm.forward_process(x_start.to('cuda'),timestep)
        noisy_images.append(reverse_transforms(x_t.to('cpu')))
    np_img = np.hstack(noisy_images)
    imwrite(f'train_{args.noise}_sample.png',np_img)
    """
    ddpm.fit(args.epochs)
    
        

         

    



