import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataloader import CelebADataset
from network_luna_bottleneck import Luna_Net
from utils import VGGPerceptualLoss

batch_size = 8
learning_rate = 0.001
num_epochs = 20
dataset_path = "dataset/CelebA-HQ"
mask_path = "dataset/irregular_masks"
in_channels = 4
out_channels = 3
factor = 1

dataset = CelebADataset(image_dir=dataset_path, mask_dir=mask_path)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Luna_Net(in_channels=in_channels, out_channels=out_channels, factor=factor)
model.to(device)

criterion = VGGPerceptualLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        _, outputs = model(images, masks)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

model.eval()
test_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

for i, (image, mask) in enumerate(zip(images, masks)):
    if i >= 10:
        break
    inpainted_img = model(image.unsqueeze(0).to(device))
    inpainted_img = inpainted_img.squeeze(0).cpu().detach()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))  # Adjust as necessary
    plt.title("Corrupted Image")
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(inpainted_img.numpy(), (1, 2, 0)))  # Adjust as necessary
    plt.title("Inpainted Image")
    plt.show()
