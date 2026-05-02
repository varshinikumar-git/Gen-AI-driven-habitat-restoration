import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === Set Your Folders and Output Path ===
train_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/generated_train"
val_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/generated_val"
output_path = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/stitched_prediction_map_color.png"

# === Function to Load Only 100th Epoch Images ===
def load_images_for_epoch(folder, epoch):
    files = [f for f in os.listdir(folder) if f.startswith(f"{epoch}_") and f.endswith(".png")]
    return sorted([os.path.join(folder, f) for f in files], key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

# === Load Paths for the 100th Epoch ===
train_epoch_images = load_images_for_epoch(train_folder, 100)
val_epoch_images = load_images_for_epoch(val_folder, 100)

# Print paths for debugging
print(f"Train images: {train_epoch_images}")
print(f"Validation images: {val_epoch_images}")

# Combine train and validation paths
image_paths = train_epoch_images + val_epoch_images
print(f"Total image paths: {image_paths}")

# === Updated Assertion for Image Count ===
assert len(image_paths) == 16, f"Expected 16 tiles (12 from train, 4 from val), found {len(image_paths)}."

# === Stitch Parameters ===
grid_rows, grid_cols = 4, 4  # 4x4 tiles
crop_border = 2              # Trim small border to reduce seams
pred_tiles = []

# === Process Each Prediction Tile ===
for path in image_paths:
    img = Image.open(path)
    w, h = img.size
    tile_w = w // 3

    # Extract the prediction slice (middle third)
    pred_img = img.crop((tile_w, 0, tile_w * 2, h))
    pred_array = np.array(pred_img)
    
    # Crop borders slightly
    cropped = pred_array[crop_border:-crop_border, crop_border:-crop_border]

    # Convert to grayscale if it's RGB
    if cropped.ndim == 3 and cropped.shape[2] == 3:
        gray = np.mean(cropped, axis=2)
    else:
        gray = cropped

    # Apply colormap for better visualization
    colored = plt.cm.viridis(gray / 255.0)[:, :, :3]  # Drop alpha
    colored_uint8 = (colored * 255).astype(np.uint8)

    pred_tiles.append(colored_uint8)

# === Stitch the Prediction Tiles ===
rows = []
for i in range(grid_rows):
    row_tiles = pred_tiles[i * grid_cols:(i + 1) * grid_cols]
    row = np.hstack(row_tiles)
    rows.append(row)

stitched_map = np.vstack(rows)

# === Save the Final Stitched Map ===
stitched_image = Image.fromarray(stitched_map)
stitched_image.save(output_path)

print(f"✅ Color-enhanced stitched prediction map saved at: {output_path}")
