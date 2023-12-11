import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from dataloader import CelebADataset
from network_luna_bottleneck import Luna_Net
from discriminator import Discriminator
from loss import CombinedLoss

batch_size = 8
learning_rate = 2 * 10e-4
num_epochs = 20
dataset_path = "dataset/CelebA-HQ"
mask_path = "dataset/irregular_masks"
in_channels = 4
out_channels = 3
factor = 8

dataset = CelebADataset(image_dir=dataset_path, mask_dir=mask_path)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_images, test_masks = next(iter(test_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

gen = Luna_Net(in_channels=in_channels, out_channels=out_channels, factor=factor)
gen.to(device)

disc = Discriminator()
disc.to(device)

criterion = CombinedLoss().to(device)
optimizer_G = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=learning_rate)

# Checkpoints for saving our ass
checkpoint_path = "latest_checkpoint.pth"
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    gen.train()
    cumulative_time = 0
    for i, (images, masks) in enumerate(train_loader):
        start_time = time.time()
        images, masks = images.to(device), masks.to(device)

        _, outputs = gen(images, masks)

        # Generator
        discriminator_output_on_generated = disc(outputs)

        generator_loss = criterion(
            outputs, images, discriminator_output_on_generated, True
        )

        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

        # Discriminator
        discriminator_output_on_real = disc(images)
        discriminator_real_loss = criterion.gan_loss(discriminator_output_on_real, True)

        # Discriminator loss for fake (generated) images
        discriminator_output_on_generated = disc(
            outputs.detach()
        )  # detach to avoid backprop to generator
        discriminator_fake_loss = criterion.gan_loss(
            discriminator_output_on_generated, False
        )

        # Total discriminator loss
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        optimizer_D.zero_grad()
        discriminator_loss.backward()
        optimizer_D.step()

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        cumulative_time += elapsed_time
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Gen Loss: {generator_loss.item():.4f},\
                Disc Loss: {discriminator_loss.item():.4f}, Time Elapsed: {cumulative_time:.2f} mins"
            )
            cumulative_time = 0

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(
            {
                "epoch": epoch,
                "gen_state_dict": gen.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
            },
            checkpoint_path,
        )

    if (epoch + 1) % 5 == 0:
        gen.eval()
        with torch.no_grad():
            for i, (image, mask) in enumerate(zip(test_images, test_masks)):
                if i >= 10:
                    break
                inpainted_img = gen(image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
                inpainted_img = inpainted_img.squeeze(0).cpu().detach()
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
                plt.title("Corrupted Image")
                plt.subplot(1, 2, 2)
                plt.imshow(np.transpose(inpainted_img.numpy(), (1, 2, 0)))
                plt.title("Inpainted Image")
                plt.savefig(f"epoch_{epoch+1}_image_{i}.png")
                plt.close()

gen.eval()
test_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = gen(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")
