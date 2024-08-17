import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np
import imageio
from skimage import io
import os
from LatentDiffusionModels import LatentDiffusion
import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA
from dataset import ImageDataset
import matplotlib.pyplot as plt
from EMA import EMA
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--train",type=str,default="")
    parser.add_argument("--val",type=str,default="")
    parser.add_argument("--save_dir",type=str,default="")
    parser.add_argument("--noise",type=str,default="linear")
    parser.add_argument("--log_dir",type=str,default="")
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--timesteps",type=int,default=300)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    IMG_SIZE=256
    BATCH_SIZE=8
    CROP_SIZE=256

    transforms = [
        KA.RandomCrop((2*CROP_SIZE,2*CROP_SIZE)),
        KA.Resize((CROP_SIZE,CROP_SIZE),antialias=True),
        KA.RandomVerticalFlip()
    ]

    train_dataset = ImageDataset(args.train,transforms=transforms)
    val_dataset = ImageDataset(args.train,transforms=transforms)

    model = LatentDiffusion(train_dataset,lr=args.lr,batch_size=BATCH_SIZE)

    """
    Autoencoder checking reconstruction
    image1 = train_dataset[0]
    image2 = model.autoencoder(image1.unsqueeze(0))[0].detach().cpu()
    
    plt.subplot(1,2,1)
    plt.imshow(image1.permute(1,2,0))
    plt.title("Input")
    plt.subplot(1,2,2)
    plt.imshow(image2.permute(1,2,0))
    plt.title("AutoEncoder Reconstruction")
    plt.savefig('autoencoder.png')
    """   
    os.makedirs(args.log_dir,exist_ok=True) 
    logger = TensorBoardLogger(
                save_dir=args.log_dir,
                name="experiment_1",
                version="v1.0",
                default_hp_metric=False  # To disable logging hparams by default
            )
    
    
    trainer = pl.Trainer(
        logger= logger,
        max_steps=2e5,
        callbacks=[EMA(0.9999)],
        gpus=[0]
    )
    trainer.fit(model)

    B=8
    model.cuda()
    out=model(batch_size=B,shape=(64,64),verbose=True)
    for idx in range(out.shape[0]):
        plt.subplot(1,len(out),idx+1)
        plt.imshow(out[idx].detach().cpu().permute(1,2,0))
        plt.axis('off')
