# ================================
# STEP 0: Install & Imports
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from google.colab import files

# ================================
# STEP 1: Upload your three outputs
# ================================
uploaded = files.upload()  # Upload pair1.png, pair2.png, pair3.png

# ================================
# STEP 2: Generate synthetic target map
# ================================
def create_restoration_target():
    p1 = np.array(Image.open('pair1.png').convert('L')).astype(np.float32) / 255.0
    p2 = np.array(Image.open('pair2.png').convert('L')).astype(np.float32) / 255.0
    p3 = np.array(Image.open('pair3.png').convert('L')).astype(np.float32) / 255.0

    restoration = 0.5 * p1 + 0.3 * p2 + 0.2 * p3
    restoration = (restoration - restoration.min()) / (restoration.max() - restoration.min())
    restoration_img = (restoration * 255).astype(np.uint8)
    Image.fromarray(restoration_img).save('restoration_target.png')
    print("✅ Saved synthetic target as 'restoration_target.png'")

create_restoration_target()

# ================================
# STEP 3: Define U-Net (No Pretrained!)
# ================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_block2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1)
        return torch.sigmoid(self.out(u1))

# ================================
# STEP 4: Dataset class
# ================================
class RestorationDataset(Dataset):
    def __init__(self):
        self.input_imgs = [Image.open(f'pair{i+1}.png').convert('L') for i in range(3)]
        self.target_img = Image.open('restoration_target.png').convert('L')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        inputs = [np.array(img).astype(np.float32)/255.0 for img in self.input_imgs]
        input_stack = np.stack(inputs, axis=0)
        target = np.array(self.target_img).astype(np.float32)/255.0
        return torch.tensor(input_stack), torch.tensor(target).unsqueeze(0)

# ================================
# STEP 5: Training
# ================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = RestorationDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True)

print("🚀 Starting training...")
for epoch in range(1000):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# ================================
# STEP 6: Save Prediction
# ================================
import matplotlib.pyplot as plt
import os

# Make sure Google Drive is mounted
from google.colab import drive
drive.mount('/content/drive')

# Set your drive save path
drive_save_path = '/content/drive/My Drive/restoration_prediction_colored.png'

# Inference and save with colormap
model.eval()
with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        pred = model(x).cpu().squeeze().numpy()
        plt.imsave(drive_save_path, pred, cmap='viridis')  # Change 'viridis' to any colormap you prefer
        print(f"✅ Colored prediction saved to: {drive_save_path}")


