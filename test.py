from model import UNET
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

def test_fn(model, image_dir, height, width, device):
    transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=[0.0, 0.0, 0.0], 
                    std=[1.0, 1.0, 1.0], 
                    max_pixel_value=255.0),
        ToTensorV2()
    ])

    image = np.array(Image.open(image_dir).convert("RGB"))
    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    model.eval()

    with torch.inference_mode():
        prediction = model(image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
        prediction = prediction.squeeze().cpu().numpy() # Remove batch dimension and move to CPU

        return prediction

        
