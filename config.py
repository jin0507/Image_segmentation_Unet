import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True # load the model from the checkpoint 
TRAIN_IMG_DIR = "data/carvana-image-masking-challenge/train/images"
TRAIN_MASK_DIR = "data/carvana-image-masking-challenge/train/masks"
VAL_IMG_DIR = "data/carvana-image-masking-challenge/val/images"
VAL_MASK_DIR = "data/carvana-image-masking-challenge/val/masks"

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0), # rotate the image by 35 degrees with a probability of 1
    A.HorizontalFlip(p=0.5), # flip the image horizontally with a probability of 0.5
    A.VerticalFlip(p=0.1), # flip the image vertically with a probability of 0.1
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])