import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

from skimage import io
import os


import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

class ImageDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transforms=None,
                 paired=True,
                 return_pair=False):
        self.root_dir = root_dir
        self.transforms = transforms
        self.paired = paired
        self.return_pair = return_pair

        if self.transforms is not None:
            if self.paired:
                data_keys = 2*['input']
            else:
                data_keys = ['input']
            
            self.input_T = KA.container.AugmentationSequential(
                *self.transforms,
                data_keys =data_keys,
                same_on_batch=False
            )
        supported_formats=['webp','jpg']        
        self.files=[el for el in os.listdir(self.root_dir) if el.split('.')[-1] in supported_formats]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        image = image_to_tensor(io.imread(img_name))/255

        if self.paired:
            c,h,w=image.shape
            slice=int(w/2)
            image2=image[:,:,slice:]
            image=image[:,:,:slice]
            if self.transforms is not None:
                out = self.input_T(image,image2)
                image=out[0][0]
                image2=out[1][0]
        elif self.transforms is not None:
            image = self.input_T(image)[0]

        if self.return_pair:
            return image2,image
        else:
            return image

