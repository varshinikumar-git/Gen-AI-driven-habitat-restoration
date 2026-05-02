import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Folders for generated outputs (epoch 100 or final saved outputs)
train_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/generated_train"
val_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/generated_val"
output_path = "/nfsshare/c2321/varshini/pix2pix_checkpoints_pair2/stitched_prediction_map_color.png"

# Helper to sort image files numerically
def load_images_sorted(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    return sorted([os.path.join(folder, f) for f in files], key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1].split(".")[0]))

# Load predicted image paths from both train and val folders
image_paths = load_images_sorted(train_folder) + load_images_sorted(val_folder)
assert len(image_paths) == 16, f"Expected 16 tiles, found {len(image_paths)}."

# Set up grid shape and processing parameters
grid_rows, grid_cols = 4, 4
crop_border = 2
pred_tiles = []

for path in image_paths:
    img = Image.open(path)
    w, h = img.size
    tile_w = w // 3
    pred_img = img.crop((tile_w, 0, tile_w * 2, h))  # predicted part (middle)

    pred_array = np.array(pred_img)
    cropped = pred_array[crop_border:-crop_border, crop_border:-crop_border]

    # Convert RGB to grayscale if needed
    if cropped.ndim == 3 and cropped.shape[2] == 3:
        gray = np.mean(cropped, axis=2)
    else:
        gray = cropped

    # Apply colormap for better visualization
    colored = plt.cm.viridis(gray / 255.0)[:, :, :3]
    colored_uint8 = (colored * 255).astype(np.uint8)

    pred_tiles.append(colored_uint8)

# Stitch all tiles into final grid
rows = []
for i in range(grid_rows):
    row_tiles = pred_tiles[i * grid_cols:(i + 1) * grid_cols]
    row = np.hstack(row_tiles)
    rows.append(row)

stitched_map = np.vstack(rows)
stitched_image = Image.fromarray(stitched_map)
stitched_image.save(output_path)

print(f"Color-enhanced stitched prediction map saved at: {output_path}")
