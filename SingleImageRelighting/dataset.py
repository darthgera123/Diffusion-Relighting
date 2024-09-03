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
    parser.add_argument("--relit_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--env_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/envmaps/med_exr")
    parser.add_argument("--diffuse_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--mask_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    parser.add_argument("--normal_path",default="/scratch/inf0/user/pgera/FlashingLights/SingleRelight/sunrise_pullover/pose_01")
    args = parser.parse_args()
    return args

def norm_img(image):
    min_value = 0.0
    max_value = 1.0

    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img.astype('float64')

def norm_env_int(env):
    energy = np.sum(env)
    img = (env-env.min())/energy

def crop_box(image_array, crop_x, crop_y, crop_width, crop_height):
    """
    Crop a specific box from the image array.

    Parameters:
    image_array (numpy.ndarray): The input image array.
    crop_x (int): The x-coordinate of the top-left corner of the crop box.
    crop_y (int): The y-coordinate of the top-left corner of the crop box.
    crop_width (int): The width of the crop box.
    crop_height (int): The height of the crop box.

    Returns:
    numpy.ndarray: The cropped image array.
    """
    return image_array[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
def resize_max_pool(image, pool_size=(16, 16)):
    # Assuming image shape is (height, width, channels)
    img_height, img_width, channels = image.shape
    pool_height, pool_width = pool_size
    
    # Define output dimensions
    out_height = img_height // pool_height
    out_width = img_width // pool_width
    
    image_reshaped = image.reshape(out_height, pool_height, out_width, pool_width, channels)
    
    pooled_image = np.max(image_reshaped, axis=(1, 3))
    return pooled_image

class PortraitDataset(Dataset):
    def __init__(self,relit_path,
                 env_path,
                 diffuse_path,
                 mask_path,
                 normal_path=None,
                 resize = True,
                 train=True,
                 test_count=100,
                 crop=True):
        
        self.relit_path = relit_path
        self.env_path = env_path
        self.diffuse_path = diffuse_path
        self.mask_path = mask_path
        self.normal_path = normal_path
        self.train = train
        if self.train:
            self.env_images = sorted(os.listdir(self.env_path))[:-test_count]
        else:
            self.env_images = sorted(os.listdir(self.env_path))[test_count:]
        # self.transforms = transforms
        self.crop = crop
        self.diffuse_img = imread(os.path.join(self.diffuse_path,'full_light.png'))/255.0
        self.mask_img = imread(os.path.join(self.mask_path,'mask.jpg'))/255.0
        if self.normal_path is not None:
            self.normal_img = imread(os.path.join(self.mask_path,'normal.png'))/255.0
        else:
            self.normal_img = None
        self.resize = resize

    def __len__(self):
        return len(self.env_images)
    
    def __getitem__(self,idx):
        data = {}
        name = self.env_images[idx].split('.')[0]
        relit = imread(os.path.join(self.relit_path,name+'.png'))/255.0
        h,w,c = relit.shape
        # relit = cv2.resize(relit,(w//2,h//2),cv2.INTER_AREA)
        
        envmaps = norm_img(imread(os.path.join(self.env_path,name+'.exr')))
        if self.crop:
            relit = crop_box(relit,160,0,512,512)
            diffuse_img = crop_box(self.diffuse_img,160,0,512,512)
            mask_img = crop_box(self.mask_img,160,0,512,512)
            if self.normal_img is not None:
                normal_img = crop_box(self.normal_img,160,0,512,512)
        
        data['relit'] = torch.from_numpy(relit).permute(2,0,1).float()
        data['env'] = torch.from_numpy(envmaps).permute(2,0,1).float()
        data['diffuse'] = torch.from_numpy(diffuse_img).permute(2,0,1).float()
        data['mask'] = torch.from_numpy(mask_img).permute(2,0,1).float()
        if self.normal_img is not None:
            data['normal'] = torch.from_numpy(normal_img).permute(2,0,1).float()
        data['name'] = name
        return data


if __name__ == "__main__":
    args = parse_args()
    
    dataset = PortraitDataset(relit_path=args.relit_path, env_path=args.env_path, \
                              diffuse_path=args.diffuse_path, mask_path=args.mask_path,\
                              normal_path=args.normal_path,train=False,crop=True)    

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1,drop_last=True)
    
    for i, data in enumerate(dataloader):
        relit = data['relit'][0].permute(1,2,0).numpy()
        relit = (relit*255).astype('uint8')
        imwrite('relit.png',relit)
        diffuse = data['diffuse'][0].permute(1,2,0).numpy()
        diffuse = (diffuse*255).astype('uint8')
        imwrite('diffuse.png',diffuse)
        normal = data['normal'][0].permute(1,2,0).numpy()
        normal = (normal*255).astype('uint8')
        imwrite('normal.png',normal)
        exit()