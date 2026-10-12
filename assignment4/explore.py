import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from data import get_raw_datasets
from data import get_loaders

# 2. Explore the Dataset

train_dataset, test_dataset = get_raw_datasets()

print("Number of Training Samples:", len(train_dataset))
print("Number of Test Samples:", len(test_dataset), "\n\n")

image, (label, mask) = train_dataset[0]
print("Dimensions of Original Image: ", image.size)
print("Image Classification: ", label)
print("Mask Dimensions: ", mask.size)
print("Mask Unique Values: ", np.unique(np.array(mask)), "\n\n")


# 3. Visualize Sample Images

# Create a figure with 3 rows and 2 columns
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i in range(3):
    image, (label, mask) = train_dataset[i]

    # Display the original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Training Image {i + 1}")
    axes[i, 0].axis("off")

    # Display the segmentation mask
    axes[i, 1].imshow(mask, cmap="viridis", vmin=1, vmax=3)
    axes[i, 1].set_title(f"Segmentation Mask {i + 1}")
    axes[i, 1].axis("off")

plt.savefig("first_three_images_and_masks.png", dpi=150, bbox_inches="tight")


# 4. Inspect one processed batch

train_loader, test_loader = get_loaders(batch_size=16)

images, (labels, masks) = next(iter(train_loader))

print("\nImages shape:", images.shape)
print("Labels shape:", labels.shape)
print("Masks shape:", masks.shape, "\n")