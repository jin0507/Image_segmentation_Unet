import cv2
import torch 
from torch import nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from test import test_fn
import config
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        target = target.float().unsqueeze(1).to(device=config.DEVICE)

        # forward pass
        with torch.cuda.amp.autocast(): # mixed precision training, use for the forward pass to speed up the training
            predictions = model(data)
            loss = loss_fn(predictions, target)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward() # scaled loss, backward pass, calculate gradients
        scaler.step(optimizer) # update optimizer parameters 
        scaler.update() # update the scaler for the next iteration

        # update tqdm lopp
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    
    model = UNET(in_channels=3, out_channels=1).to(device=config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # binary cross entropy loss function for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  

    train_loader, val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, 
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        train_transform,
        val_transform,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=config.DEVICE) # check accuracy before training, after it calculate the accuracy then back to the training loop
    scaler = torch.cuda.amp.GradScaler() # mixed precision training, use for the backward pass to speed up the training
    # scaler used to scale the loss value, then call the backward pass, then call the optimizer step, then update the scaler for the next iteration

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader=train_loader, 
                 model=model, 
                 optimizer=optimizer, 
                 loss_fn=loss_fn, 
                 scaler=scaler)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        # Check acuracy with the validation set
        check_accuracy(val_loader, model, device=config.DEVICE)

        # print sone examples to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=config.DEVICE)

if __name__ == "__main__":
    main()
    # model = UNET(in_channels=3, out_channels=1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # if config.LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #     Test the model with a single image
    #     test_image_path = "D:/JIN/image_segmentation/semantic_segmentation_unet/OIP.jfif"
    #     prediction = test_fn(model=model, 
    #                          image_dir=test_image_path, 
    #                          height=config.IMAGE_HEIGHT, 
    #                          width=config.IMAGE_WIDTH, 
    #                          device=device)

    #     Load the target mask for comparison
    #     target_path = "data/carvana-image-masking-challenge/val/masks/fff9b3a5373f_12_mask.gif"
    #     target = np.array(Image.open(target_path).convert("L"))
    #     target = cv2.resize(target, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    #     Load the original image for visualization
    #     original = np.array(Image.open(test_image_path).convert("RGB"))
    #     original = cv2.resize(original, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    #     original[prediction > 0] = (0, 255, 0)  # Highlight the predicted mask in green

    #     Visualize the original image, target mask, and predicted mask
    #     cv2.imshow("Mask on Original", original)
    #     cv2.imshow("Target Mask", target)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
