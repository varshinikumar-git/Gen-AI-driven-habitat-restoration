from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the images
pred_path = "/nfsshare/c2321/varshini/pix2pix_checkpoints/stitched_prediction_map.png"
forest_loss_path = "/nfsshare/c2321/varshini/forest_loss.png"
land_cover_loss_path = "/nfsshare/c2321/varshini/land_cover_loss.png"

# Convert to grayscale for comparison purposes
pred_img = Image.open(pred_path).convert("L").resize((256, 256))
forest_loss_img = Image.open(forest_loss_path).convert("L").resize((256, 256))
land_cover_loss_img = Image.open(land_cover_loss_path).convert("L").resize((256, 256))

# Convert to numpy arrays and normalize to [0, 1]
pred_np = np.array(pred_img) / 255.0
forest_np = np.array(forest_loss_img) / 255.0
landcover_np = np.array(land_cover_loss_img) / 255.0

# Thresholding to get binary degradation maps
pred_mask = pred_np > 0.5
forest_mask = forest_np > 0.5
landcover_mask = landcover_np > 0.5

# Composite map encoding:
# 0: no degradation
# 1: prediction only (early signal)
# 2: matches forest loss only
# 3: matches land cover loss only
# 4: matches both forest & land cover loss
# 5: prediction + forest loss
# 6: prediction + land cover loss
# 7: all three overlap

composite_map = (
    (pred_mask.astype(int) * 1) + 
    (forest_mask.astype(int) * 2) + 
    (landcover_mask.astype(int) * 4)
)

# Create a colormap for visualization
cmap = plt.cm.get_cmap('tab10', 8)
labels = [
    "No degradation",
    "Predicted only",
    "Forest loss only",
    "Forest + Predicted",
    "Land cover loss only",
    "Land cover + Predicted",
    "Forest + Land cover",
    "All three overlap"
]

# Plot
plt.figure(figsize=(10, 6))
im = plt.imshow(composite_map, cmap=cmap, vmin=0, vmax=7)
cbar = plt.colorbar(im, ticks=range(8))
cbar.ax.set_yticklabels(labels)
plt.title("Composite Degradation Map")
plt.axis('off')
plt.tight_layout()
plt.show()
