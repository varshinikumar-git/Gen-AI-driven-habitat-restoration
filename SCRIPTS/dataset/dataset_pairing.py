import os
from PIL import Image
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Root directory for data
ROOT_DIR = "/IBMSTORAGE/users/bair23/.ssh/"

# Resize dimensions and dataset pair definitions
TILE_SIZE = (256, 256)
PAIRS = [
    ("land_cover_loss.png", "forest_loss.png", "pix2pix_dataset_pair1"),
    ("soil_carbon.png", "water_occurrence.png", "pix2pix_dataset_pair2"),
    ("mean_temperature.png", "forest_loss.png", "pix2pix_dataset_pair3"),
]

def split_image_to_tiles(img, tile_size):
    """Split a large image into non-overlapping tiles of given size."""
    width, height = img.size
    tiles = []
    for top in range(0, height, tile_size[1]):
        for left in range(0, width, tile_size[0]):
            box = (left, top, left + tile_size[0], top + tile_size[1])
            tile = img.crop(box)
            if tile.size == tile_size:
                tiles.append(tile)
    return tiles

def prepare_dataset_pairs():
    for input_name, target_name, output_dir in PAIRS:
        logging.info(f"Preparing pair: {input_name} → {target_name}")

        input_path = os.path.join(ROOT_DIR, input_name)
        target_path = os.path.join(ROOT_DIR, target_name)

        # Load and resize images
        try:
            input_img = Image.open(input_path).convert("RGB").resize((1024, 1024))
            target_img = Image.open(target_path).convert("RGB").resize((1024, 1024))
        except FileNotFoundError as e:
            logging.error(f"File not found: {e.filename}")
            continue

        # Split into tiles
        input_tiles = split_image_to_tiles(input_img, TILE_SIZE)
        target_tiles = split_image_to_tiles(target_img, TILE_SIZE)

        # Prepare output directories
        train_dir = os.path.join(ROOT_DIR, output_dir, "train")
        val_dir = os.path.join(ROOT_DIR, output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Save paired images
        total = len(input_tiles)
        train_cutoff = int(0.8 * total)
        for i, (inp, tgt) in tqdm(enumerate(zip(input_tiles, target_tiles)), total=total, desc=output_dir):
            paired = Image.new("RGB", (TILE_SIZE[0] * 2, TILE_SIZE[1]))
            paired.paste(inp, (0, 0))
            paired.paste(tgt, (TILE_SIZE[0], 0))

            save_path = os.path.join(train_dir if i < train_cutoff else val_dir, f"{i}.png")
            paired.save(save_path)

        logging.info(f"✅ Finished preparing: {output_dir}")

if __name__ == "__main__":
    prepare_dataset_pairs()
    logging.info("✅ All dataset pairs prepared successfully.")
