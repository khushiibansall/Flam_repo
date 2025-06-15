import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('indoor.jpg')
if img is None:
    print("Error: Could not load indoor.jpg")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 11), 0)

grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

magnitude = np.sqrt(grad_x**2 + grad_y**2)
direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)

weighted_dir = direction * magnitude
mean_dir = np.sum(weighted_dir) / np.sum(magnitude)

print(f"\nEstimated Dominant Light Gradient Angle (degrees): {mean_dir:.2f}°")

plt.imshow(gray, cmap='gray')
plt.title("Brightness Map (Grayscale)")
plt.axis('off')
plt.savefig("brightness_heatmap.jpg")
print("Saved brightness heatmap as brightness_heatmap.jpg")
