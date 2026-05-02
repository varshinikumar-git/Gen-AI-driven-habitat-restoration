import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# === Paths ===
train_dir = "/nfsshare/c2321/varshini/pix2pix_dataset_pair2/train"
val_dir = "/nfsshare/c2321/varshini/pix2pix_dataset_pair2/val"
save_dir = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Dataset ===
class Pix2PixDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(os.listdir(folder))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.files[index])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        input_img = image.crop((0, 0, w // 2, h))
        target_img = image.crop((w // 2, 0, w, h))
        return self.transform(input_img), self.transform(target_img), self.files[index]

    def __len__(self):
        return len(self.files)

train_loader = DataLoader(Pix2PixDataset(train_dir), batch_size=4, shuffle=True)
val_loader = DataLoader(Pix2PixDataset(val_dir), batch_size=1, shuffle=False)

# === Models ===
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        def block(in_c, out_c, down=True, act='relu', use_dropout=False):
            layers = []
            if down:
                layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            if act == 'relu':
                layers.append(nn.ReLU(True))
            else:
                layers.append(nn.LeakyReLU(0.2, True))
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = block(in_channels, 64, down=True, act='lrelu')
        self.down2 = block(64, 128, down=True, act='lrelu')
        self.down3 = block(128, 256, down=True, act='lrelu')
        self.down4 = block(256, 512, down=True, act='lrelu')
        self.up1 = block(512, 256, down=False)
        self.up2 = block(512, 128, down=False)
        self.up3 = block(256, 64, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        return self.final(torch.cat([u3, d1], 1))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        def disc_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )
        self.model = nn.Sequential(
            disc_block(in_channels, 64),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# === Training Setup ===
G = UNetGenerator().to(device)
D = PatchGANDiscriminator().to(device)
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
scheduler_G = StepLR(optimizer_G, step_size=20, gamma=0.5)
scheduler_D = StepLR(optimizer_D, step_size=20, gamma=0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5,), (0.5,))
])

# === Save Outputs ===
def save_generated_outputs(G, folder, output_folder, epoch):
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(os.listdir(folder))
    for file in files:
        path = os.path.join(folder, file)
        image = Image.open(path).convert("RGB")
        w, h = image.size
        input_img = image.crop((0, 0, w // 2, h))
        target_img = image.crop((w // 2, 0, w, h))

        input_tensor = transform(input_img).unsqueeze(0).to(device)
        target_tensor = transform(target_img).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_tensor = G(input_tensor)

        comparison = torch.cat([input_tensor, fake_tensor, target_tensor], dim=3)
        utils.save_image(comparison * 0.5 + 0.5, os.path.join(output_folder, f"{epoch:03d}_{file}"))

# === Training Loop ===
num_epochs = 100
for epoch in range(num_epochs):
    G.train()
    D.train()
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for i, (input_img, target_img, _) in enumerate(loop):
        input_img, target_img = input_img.to(device), target_img.to(device)

        # Train Discriminator
        fake_img = G(input_img)
        real_pred = D(input_img, target_img)
        fake_pred = D(input_img, fake_img.detach())
        d_loss = (criterion_GAN(real_pred, torch.ones_like(real_pred)) +
                  criterion_GAN(fake_pred, torch.zeros_like(fake_pred))) * 0.5
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        fake_pred = D(input_img, fake_img)
        g_loss = criterion_GAN(fake_pred, torch.ones_like(fake_pred)) + 200 * criterion_L1(fake_img, target_img)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

    scheduler_G.step()
    scheduler_D.step()

    # Save model and images
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        torch.save(G.state_dict(), os.path.join(save_dir, f"gen_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(save_dir, f"disc_epoch{epoch+1}.pth"))
        save_generated_outputs(G, train_dir, os.path.join(save_dir, f"generated_train"), epoch+1)
        save_generated_outputs(G, val_dir, os.path.join(save_dir, f"generated_val"), epoch+1)
