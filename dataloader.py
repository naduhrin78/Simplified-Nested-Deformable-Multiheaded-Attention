from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import BinarizeMask


class CelebADataset(Dataset):
    def __init__(self, image_dir, mask_dir, dilation_range=(9, 49)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dilation_range = dilation_range

        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                BinarizeMask(threshold=0.6),
                transforms.RandomAffine(degrees=20, scale=(1.2, 1.3)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load a random mask
        mask = None

        while mask is None:
            mask_name = random.choice(self.masks)
            mask_path = os.path.join(self.mask_dir, mask_name)

            try:
                mask = Image.open(mask_path).convert("L")
            except:
                continue

        # Transform
        seed = np.random.randint(2147483647)
        random.seed(seed)
        image = self.image_transform(image)
        random.seed(seed)
        mask = self.mask_transform(mask)

        corrupted_image = image * mask

        return corrupted_image, image, mask


def show_image(image, title="Masked Image"):
    """Helper function to display a single image."""
    # Convert image tensor to numpy and denormalize
    image = image.numpy().transpose((1, 2, 0))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset = CelebADataset(
        image_dir="dataset/CelebA-HQ", mask_dir="dataset/irregular_masks"
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    image, _ = next(iter(dataloader))
    show_image(image[0])
