import os
import shutil
import random

# === UPDATE THESE ===
input_folder = r"D:\genai_datasets\final_datasets"   # <-- Folder with land cover loss
target_folder = r"D:\genai_datasets\final_datasets"  # <-- Folder with soil carbon
output_root = r"D:\genai_datasets\pix2pix_dataset"

# === DO NOT TOUCH BELOW UNLESS NEEDED ===
train_ratio = 0.8
random.seed(42)

# Get matched image files
input_images = sorted(os.listdir(input_folder))
target_images = sorted(os.listdir(target_folder))
matched_files = list(set(input_images) & set(target_images))

if not matched_files:
    raise ValueError("⚠️ No matching files found between input and target folders!")

# Shuffle for splitting
random.shuffle(matched_files)
split_idx = int(len(matched_files) * train_ratio)
train_files = matched_files[:split_idx]
val_files = matched_files[split_idx:]

# Define Pix2Pix folder structure
for phase in ["train", "val"]:
    for domain in ["A", "B"]:
        os.makedirs(os.path.join(output_root, phase, domain), exist_ok=True)

# Copy files into the proper folders
def copy_files(file_list, phase):
    for fname in file_list:
        shutil.copy(
            os.path.join(input_folder, fname),
            os.path.join(output_root, phase, "A", fname)
        )
        shutil.copy(
            os.path.join(target_folder, fname),
            os.path.join(output_root, phase, "B", fname)
        )

copy_files(train_files, "train")
copy_files(val_files, "val")

print(f"✅ Pix2Pix dataset ready at: {output_root}")
print(f"   ➤ Train: {len(train_files)} pairs")
print(f"   ➤ Val:   {len(val_files)} pairs")
