import os
from PIL import Image

# Define root path
root_dir = "/nfsshare/c2321/varshini/"

# Define pixel size
resize_to = (256, 256)

# Define dataset pairs
pairs = [
    ("land_cover_loss.png", "forest_loss.png", "pix2pix_dataset_pair1"),
    ("soil_carbon.png", "water_occurrence.png", "pix2pix_dataset_pair2"),
    ("mean_temperature.png", "forest_loss.png", "pix2pix_dataset_pair3"),
]

# Function to split a big image into 256x256 tiles
def split_image_to_tiles(img, tile_size):
    width, height = img.size
    tiles = []
    for top in range(0, height, tile_size[1]):
        for left in range(0, width, tile_size[0]):
            box = (left, top, left + tile_size[0], top + tile_size[1])
            tile = img.crop(box)
            if tile.size == tile_size:
                tiles.append(tile)
    return tiles

for input_file, target_file, output_dir in pairs:
    print(f"Preparing pair: {input_file} → {target_file}")
    
    input_path = os.path.join(root_dir, input_file)
    target_path = os.path.join(root_dir, target_file)
    
    # Load images
    input_img = Image.open(input_path).convert("RGB").resize((1024, 1024))
    target_img = Image.open(target_path).convert("RGB").resize((1024, 1024))

    # Split into tiles
    input_tiles = split_image_to_tiles(input_img, resize_to)
    target_tiles = split_image_to_tiles(target_img, resize_to)

    # Create output dirs
    train_dir = os.path.join(root_dir, output_dir, "train")
    val_dir = os.path.join(root_dir, output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Save 80% train, 20% val
    total = len(input_tiles)
    train_cutoff = int(0.8 * total)
    for i, (inp, tgt) in enumerate(zip(input_tiles, target_tiles)):
        paired = Image.new("RGB", (resize_to[0] * 2, resize_to[1]))
        paired.paste(inp, (0, 0))
        paired.paste(tgt, (resize_to[0], 0))

        if i < train_cutoff:
            paired.save(os.path.join(train_dir, f"{i}.png"))
        else:
            paired.save(os.path.join(val_dir, f"{i}.png"))

print("✅ All dataset pairs prepared successfully.")
