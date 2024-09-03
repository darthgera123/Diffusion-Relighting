import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import  transforms
from argparse import ArgumentParser
from imageio.v2 import imread,imwrite
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PortraitDataset
from cunet import CondUnet
from PerceptualSimilarity.lpips.lpips import lpips



def to_cuda(pred):
    
    params = ['diffuse','relit','env']
    for param in params:
        pred[param] = pred[param].cuda() 
    return pred

def to_cpu(pred):
    
    params = ['diffuse','relit','env']
    for param in params:
        pred[param] = pred[param].cpu().detach() 
    return pred

def norm_img_torch(image):
    min_value = 0.0
    max_value = 1.0

    image_min = torch.min(image)
    image_max = torch.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

class RelitTrainer:
    def __init__(self, model, train_dataloader, 
                 val_dataloader,
                 log_dir='runs/unet_experiment', 
                 save_dir='val_predictions', 
                 lr=0.001,epochs=500):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.criterion = nn.L1Loss()  
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_dir = save_dir
        self.val_dir = os.path.join(self.save_dir,'renders')
        self.model_dir = os.path.join(self.save_dir,'checkpoint')
        
        os.makedirs(save_dir,exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)
        os.makedirs(log_dir,exist_ok=True)
        os.makedirs(self.val_dir,exist_ok=True)

        self.model_name = model
        latent_size = (4,16,16)
        self.model = CondUnet(in_ch=3,out_ch=3,latent_size=latent_size)
        self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6, verbose=True)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()

    def comp_loss(self,pred,gt):
        l1 = self.criterion(pred,gt)
        vgg = self.loss_fn_vgg.forward(pred,gt)
        return l1+1e-1*vgg


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, data_idx in progress_bar:
            self.optimizer.zero_grad()
            data_idx = to_cuda(data_idx)
            relit,env,diffuse = data_idx['relit'],data_idx['env'],data_idx['diffuse']
            pred = self.model(diffuse,env)
            loss = self.comp_loss(pred,relit)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Update progress bar every iteration with the current loss
            progress_bar.set_postfix({'loss': '{:.6f}'.format(loss.item())})

            # Optionally, log to TensorBoard here as well
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_dataloader) + batch_idx)

        avg_loss = total_loss / len(self.train_dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Training Loss: {avg_loss:.6f}")

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data_idx in enumerate(self.val_dataloader):
                
                data_idx = to_cuda(data_idx)
                relit,env,diffuse = data_idx['relit'],data_idx['env'],data_idx['diffuse']
                pred = self.model(diffuse,env)
                
                
                loss = self.comp_loss(pred,relit)
                total_loss += loss.item()
                if batch_idx == 0:  # Log validation images once per epoch
                    self.writer.add_images('Validation/Input Images', env, epoch)
                    
                    
                    self.writer.add_images('Validation/Output Images', pred, epoch)
                    self.writer.add_images('Validation/Target Images', relit, epoch)
            # Log validation loss using TensorBoard
            
            self.writer.add_scalar('Loss/validation', total_loss / len(self.val_dataloader), epoch)
        self.scheduler.step(total_loss)
        print(f"Validation Loss: {total_loss / len(self.val_dataloader)}")

    
    def save_val_predictions(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data_idx in enumerate(self.val_dataloader):
                data_idx = to_cuda(data_idx)
                relit,env,diffuse = data_idx['relit'],data_idx['env'],data_idx['diffuse']
                pred = self.model(diffuse,env)
                relit = relit.detach().cpu()
                pred = pred.detach().cpu()
                diffuse = diffuse.detach().cpu()
                for i in range(env.shape(0)):

                    diffuse_img = transforms.ToPILImage()(diffuse[i]).convert("RGB")
                    relit_img = transforms.ToPILImage()(relit[i]).convert("RGB")
                    pred_img = transforms.ToPILImage()(pred[i]).convert("RGB")

                    diffuse_np = np.array(diffuse_img)
                    relit_np = np.array(relit_img)
                    pred_np = np.array(pred_img)
                    combined_img = np.clip(np.hstack(diffuse_np,pred_np,relit_np))
                    name = data_idx['name'][i]

                    file_name = os.path.join(self.val_dir, f'epoch_{epoch}_{name}.png')    
                    imwrite(file_name, combined_img)
                return
    
    def save_checkpoint(self, epoch):
        path = os.path.join(self.model_dir,f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

    def save_predictions(self,dataloader,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch_idx, data_idx in enumerate(tqdm(dataloader)):
                data_idx = to_cuda(data_idx)
                relit,env,diffuse = data_idx['relit'],data_idx['env'],data_idx['diffuse']
                pred = self.model(diffuse,env)
                relit = relit.detach().cpu()
                pred = pred.detach().cpu()
                diffuse = diffuse.detach().cpu()
                for i in range(env.shape(0)):

                    diffuse_img = transforms.ToPILImage()(diffuse[i]).convert("RGB")
                    relit_img = transforms.ToPILImage()(relit[i]).convert("RGB")
                    pred_img = transforms.ToPILImage()(pred[i]).convert("RGB")

                    diffuse_np = np.array(diffuse_img)
                    relit_np = np.array(relit_img)
                    pred_np = np.array(pred_img)
                    combined_img = np.clip(np.hstack(diffuse_np,pred_np,relit_np))
                    name = data_idx['name'][i]

                    file_name = os.path.join(save_dir, f'{name}.png')    
                    imwrite(file_name, combined_img)
                

    def fit(self, epochs, validate_every_n_epochs=20,save_model =100):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % validate_every_n_epochs == 0:
                self.validate(epoch)
                self.save_val_predictions(epoch)  # Save validation predictions on specified epochs
            if (epoch + 1) % save_model == 0:
                self.save_checkpoint(epoch)  # Save validation predictions on specified epochs
                
        self.writer.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--relit_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--env_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/envmaps/med_exr")
    parser.add_argument("--diffuse_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--mask_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--normal_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--save_dir", default="transforms.json", help="output path")
    parser.add_argument("--checkpoint", default="transforms.json", help="output path")
    parser.add_argument("--lr",type=float, default=1e-3, help="output path")
    parser.add_argument("--epochs",type=int,default=100, help="create_images")
    parser.add_argument("--train",action='store_true', help="create_images")
    parser.add_argument("--network", default="uvrelit", help="output path")
    args = parser.parse_args()

    uv_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    env_transform = transforms.Compose([
            transforms.Resize((16, 32)),  # Downsample the envmap
            transforms.ToTensor(),
        ])
    if args.network == 'latent':
        resize = False
    else:
        resize = True
    train_data = PortraitDataset(relit_path=args.relit_path, env_path=args.env_path, \
                              diffuse_path=args.diffuse_path, mask_path=args.mask_path,\
                              normal_path=args.normal_path,crop=True)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1,drop_last=True)
    val_data = PortraitDataset(relit_path=args.relit_path, env_path=args.env_path, \
                              diffuse_path=args.diffuse_path, mask_path=args.mask_path,\
                              normal_path=args.normal_path,train=False,crop=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=1,drop_last=True)
    log_dir = os.path.join(args.save_dir,'logs')
    
    
    trainer = RelitTrainer(args.network, train_dataloader, val_dataloader,save_dir=args.save_dir,log_dir=log_dir,lr=args.lr)
    if args.train:
        trainer.fit(epochs=args.epochs,validate_every_n_epochs=20)
        val_save_path = os.path.join(args.save_dir,'val')
        trainer.save_predictions(val_dataloader,val_save_path)
        if not args.skip_train:
            train_save_path = os.path.join(args.save_dir,'train')
            trainer.save_predictions(train_dataloader,train_save_path)
    else:
        trainer.load_checkpoint(args.checkpoint)
        test_data = PortraitDataset(relit_path=args.relit_path, env_path=args.env_path, \
                              diffuse_path=args.diffuse_path, mask_path=args.mask_path,\
                              normal_path=args.normal_path,train=False,crop=True)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        test_save_path = os.path.join(args.save_dir,'val')
        train_save_path = os.path.join(args.save_dir,'train')
        # trainer.save_predictions(test_dataloader,test_save_path)
        if not args.skip_train:
            trainer.save_predictions(train_dataloader,train_save_path)

    