import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('indoor.jpg')
if img is None:
    print("Error: Could not load indoor.jpg")
    exit()

# Convert to grayscale (brightness proxy)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smooth the image to remove noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Compute the gradients (x and y directions)
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# Compute the gradient magnitude and direction
magnitude = np.sqrt(grad_x**2 + grad_y**2)
direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)

# Find the average gradient direction (weighted by magnitude)
weighted_dir = direction * magnitude
mean_dir = np.sum(weighted_dir) / np.sum(magnitude)

print(f"\nEstimated Dominant Light Gradient Angle (degrees): {mean_dir:.2f}°")

# Optional: Visualize the brightness heatmap
plt.imshow(gray, cmap='gray')
plt.title("Brightness Map (Grayscale)")
plt.axis('off')
plt.savefig("brightness_heatmap.jpg")
print("Saved brightness heatmap as brightness_heatmap.jpg")
