from pathlib import Path
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

image_dir = Path('data/carvana-image-masking-challenge/images')
mask_dir = Path('data/carvana-image-masking-challenge/masks')

def split_data(image_dir, mask_dir):
    # Create a list of all images and masks
    images = list(image_dir.glob("*.jpg"))  # Assuming images are in .jpg format
    masks = list(mask_dir.glob("*.gif"))    # Assuming masks are in .gif format

    # Split the data into 80% train, 20% validation
    train_size = int(len(images) * 0.8)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Create directories for train and validation sets
    train_image_dir = Path("data/carvana-image-masking-challenge/train/images")
    train_mask_dir = Path("data/carvana-image-masking-challenge/train/masks")
    val_image_dir = Path("data/carvana-image-masking-challenge/val/images")
    val_mask_dir = Path("data/carvana-image-masking-challenge/val/masks")

    train_image_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)

    # Move images and masks to train directory
    for img in train_images:
        try:
            shutil.move(str(img), str(train_image_dir / img.name))
            mask = mask_dir / (img.stem + "_mask.gif")
            if mask.exists():
                shutil.move(str(mask), str(train_mask_dir / mask.name))
        except Exception as e:
            logging.error(f"Error moving file {img}: {e}")

    # Move images and masks to validation directory
    for img in val_images:
        try:
            shutil.move(str(img), str(val_image_dir / img.name))
            mask = mask_dir / (img.stem + "_mask.gif")
            if mask.exists():
                shutil.move(str(mask), str(val_mask_dir / mask.name))
        except Exception as e:
            logging.error(f"Error moving file {img}: {e}")

    return train_image_dir, train_mask_dir, val_image_dir, val_mask_dir

if __name__ == "__main__":
    split_data(image_dir, mask_dir)
    logging.info("Data split successfully!")