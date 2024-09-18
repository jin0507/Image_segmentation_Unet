import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir  
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # L is for grayscale
        mask[mask == 255.0] = 1.0 # Since there are values with 255, we convert them to 1.0 to represent the mask

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"] # We are not using the mask here
            mask = augmentations["mask"] # We are not using the image here

        return image, mask