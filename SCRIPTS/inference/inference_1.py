import os
from PIL import Image
import numpy as np

# Folders
train_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints/generated_train"
val_folder = "/nfsshare/c2321/varshini/pix2pix_checkpoints/generated_val"
output_path = "/nfsshare/c2321/varshini/pix2pix_checkpoints/stitched_prediction_map.png"

# Combine and sort image paths based on numeric index
def load_images_sorted(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    return sorted([os.path.join(folder, f) for f in files], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

image_paths = load_images_sorted(train_folder) + load_images_sorted(val_folder)  # 0–15
assert len(image_paths) == 16, f"Expected 16 tiles, found {len(image_paths)}."

# Stitch parameters
grid_rows, grid_cols = 4, 4
pred_tiles = []

# Extract middle (prediction) from each image
for path in image_paths:
    img = Image.open(path)
    w, h = img.size
    tile_w = w // 3
    pred_img = img.crop((tile_w, 0, tile_w * 2, h))  # middle third
    pred_tiles.append(np.array(pred_img))

# Stitch into full map
rows = []
for i in range(grid_rows):
    row_tiles = pred_tiles[i * grid_cols:(i + 1) * grid_cols]
    row = np.hstack(row_tiles)
    rows.append(row)

stitched_map = np.vstack(rows)
stitched_image = Image.fromarray(stitched_map)
stitched_image.save(output_path)

print(f"✅ Full stitched prediction map saved at: {output_path}")
