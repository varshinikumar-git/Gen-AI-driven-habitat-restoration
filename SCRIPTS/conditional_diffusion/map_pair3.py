import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import random
from google.colab import drive
drive.mount('/content/drive')

# --------- UNET-Like Conditional Denoiser ---------
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=64):
        super(ConditionalUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# --------- Load and Preprocess Image and Mask ---------
def load_image(image_path, resize=256):
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])
    return tf(img).unsqueeze(0)

def generate_grid_mask(image_tensor, patch_size=64, line_thickness=2):
    _, _, h, w = image_tensor.shape
    mask = torch.zeros_like(image_tensor[:, :1, :, :])  # Single channel
    for y in range(0, h, patch_size):
        mask[:, :, y:y+line_thickness, :] = 1.0
    for x in range(0, w, patch_size):
        mask[:, :, :, x:x+line_thickness] = 1.0
    return mask

# --------- Training Loop ---------
def train(model, image_with_grid, mask, num_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(image_with_grid, mask)
        loss = loss_fn(output, image_with_grid * (1 - mask))  # Only supervise on non-grid
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    return output

# --------- Paths and Setup ---------
input_path = "/content/drive/My Drive/pix2pix_checkpoints_pair1/stitched_predicted_map.png"
os.makedirs("outputs", exist_ok=True)

# --------- Main Pipeline ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
input_img = load_image(input_path, resize=256).to(device)
grid_mask = generate_grid_mask(input_img, patch_size=64).to(device)

# Initialize and train model
model = ConditionalUNet().to(device)
denoised_output = train(model, input_img, grid_mask, num_epochs=100)
# Save results locally
save_image(input_img, "outputs/input_gridlines.png")
save_image(denoised_output, "outputs/denoised_output.png")
#save_image(grid_mask, "outputs/grid_mask.png")

# --------- Save Outputs to Google Drive ---------
output_dir = "/content/drive/My Drive/pix2pix_checkpoints_pair1"
os.makedirs(output_dir, exist_ok=True)

save_image(input_img, os.path.join(output_dir, "input_gridlines.png"))
save_image(denoised_output, os.path.join(output_dir, "denoised_output.png"))
save_image(grid_mask, os.path.join(output_dir, "grid_mask.png"))

# --------- Blended Overlay Image (Joint Visualization) ---------
# This combines input and denoised output directly, not side-by-side
blended = 0.6 * input_img + 0.5 * denoised_output+ 0.3 * grid_mask
blended = torch.clamp(blended, 0.0, 1.0)

# Save blended overlay
save_image(blended, "outputs/blended_overlay.png")
save_image(blended, os.path.join(output_dir, "blended_overlay.png"))
