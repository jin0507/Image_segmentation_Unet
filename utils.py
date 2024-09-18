import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import cv2

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir,
                train_maskdir,
                val_dir,
                val_maskdir,
                batch_size,
                train_transform,
                val_transform,
                num_workers=4,
                pin_memory=True):
    
    train_dataset = CarvanaDataset(image_dir=train_dir,
                                   mask_dir=train_maskdir,
                                   transform=train_transform)
    
    val_dataset = CarvanaDataset(image_dir=val_dir,
                                  mask_dir=val_maskdir,
                                  transform=val_transform)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"): # check accuracy on training & validation to see how well the model is doing    
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad(): # no need to calculate gradients
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x)) # sigmoid to get probabilities
            preds = (preds > 0.5).float() # convert to binary values, example: 0.6 -> 1, 0.4 -> 0
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds) # numel returns the number of elements in the tensor
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train() # set model back to training mode after evaluation

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """
    Save the predictions of the model as images
    Parameters:
        loader: DataLoader - the data loader to get the images
        model: nn.Module - the model to get the predictions from
        folder: str - the folder to save the images
        device: str - the device to run the model on
    Returns: None
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train() # set model back to training mode after evaluation 

# def show_prediction(model, test_image_path, target_mask_path, device="cuda"):
#     image = Image.open(image_path)
#     image = test_transform(image=image)["image"]
#     image = image.unsqueeze(0).to(device)
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         preds = torch.sigmoid(model(image))
#         preds = (preds > 0.5).float()
#     plt.imshow(preds.squeeze().cpu(), cmap="gray")
#     plt.show()
#     model.train() # set model back to training mode after evaluation