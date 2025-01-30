import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define dataset paths
DATASET_PATH = "archive\\PennFudanPed"
IMAGE_PATH = os.path.join(DATASET_PATH, "PNGImages")
MASK_PATH = os.path.join(DATASET_PATH, "PedMasks")

# Get file list
image_files = sorted(os.listdir(IMAGE_PATH))
mask_files = sorted(os.listdir(MASK_PATH))

# Load an image and its corresponding mask
image = cv2.imread(os.path.join(IMAGE_PATH, image_files[0]))
mask = cv2.imread(os.path.join(MASK_PATH, mask_files[0]), cv2.IMREAD_GRAYSCALE)

# Display image and mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.show()
