import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from imageio.v2 import imread,imwrite
import cv2
from PIL import Image
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--images",default="")
    args = parser.parse_args()
    return args

class DiffusionDataset(Dataset):
    def __init__(self,image_path,transforms=None):
        
        self.images_path = image_path
        self.images = os.listdir(image_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.images_path,self.images[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        
        return img
    
def custom_collate(batch):
    # Ensure all elements in the batch are tensors
    batch = [torch.tensor(item, dtype=torch.float32) if not isinstance(item, torch.Tensor) else item for item in batch]
    
    # Ensure all tensors have the same shape
    shapes = [item.shape for item in batch]
    assert all(shape == shapes[0] for shape in shapes), "Inconsistent shapes in batch"
    
    # Stack tensors into a single batch
    return torch.stack(batch, dim=0)


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

    img_dataloader = DataLoader(diff_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    
    # for i,img in enumerate(img_dataloader):
    #     single = img[0,:,:,:]
    #     print(img.shape,single.shape)
    #     np_img = reverse_transforms(single)
    #     if i==1:
    #         break
    
    # print(np_img.shape)
    # imwrite('test.png',np_img)
    # image = next(iter(img_dataloader))[0]
    # np_img = reverse_transforms(image)
    # imwrite('test.png',np_img)
