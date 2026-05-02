import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
tile_size = 256
input_image_path = 'final_image_1.png'

# Target maps for different prediction tasks
target_maps = {
    'land_cover_loss': 'final_image_2.png',
    'mean_temp': 'final_image_3.png',
    'soil_carbon': 'final_image_4.png',
    'water_occurrence': 'final_image_5.png'
}

output_root = 'pix2pix_dataset'
val_ratio = 0.2


def split_and_save_tiles(input_img, target_img, task_name):
    if input_img.shape != target_img.shape:
        print(f"[❌] Shape mismatch for '{task_name}': {input_img.shape} vs {target_img.shape}")
        return

    h, w, _ = input_img.shape
    if h < tile_size or w < tile_size:
        print(f"[❌] Image too small for tiling: {w}x{h}. Skipping '{task_name}'")
        return

    tiles_input = []
    tiles_target = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if y + tile_size <= h and x + tile_size <= w:
                inp_tile = input_img[y:y + tile_size, x:x + tile_size]
                tgt_tile = target_img[y:y + tile_size, x:x + tile_size]
                tiles_input.append(inp_tile)
                tiles_target.append(tgt_tile)

    if len(tiles_input) == 0:
        print(f"[❌] No valid tiles could be extracted for '{task_name}'")
        return

    # Split into train and val
    train_input, val_input, train_target, val_target = train_test_split(
        tiles_input, tiles_target, test_size=val_ratio, random_state=42
    )

    # Output directories
    for split, split_inputs, split_targets in zip(
        ['train', 'val'], [train_input, val_input], [train_target, val_target]
    ):
        input_dir = os.path.join(output_root, task_name, split, 'input')
        target_dir = os.path.join(output_root, task_name, split, 'target')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        for i, (inp, tgt) in enumerate(zip(split_inputs, split_targets)):
            cv2.imwrite(os.path.join(input_dir, f'{task_name}_{split}_{i}.png'), inp)
            cv2.imwrite(os.path.join(target_dir, f'{task_name}_{split}_{i}.png'), tgt)

    print(f"[✓] {task_name}: {len(train_input)} training tiles, {len(val_input)} validation tiles saved.")


# === MAIN SCRIPT ===
if __name__ == '__main__':
    input_img = cv2.imread(input_image_path)
    if input_img is None:
        print(f"[❌] Could not read input image: {input_image_path}")
        exit()

    for task, target_img_path in target_maps.items():
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            print(f"[❌] Could not read target image for '{task}': {target_img_path}")
            continue

        split_and_save_tiles(input_img, target_img, task)

    print("\n✅ All datasets prepared. Ready for Pix2Pix training on HPC.")
