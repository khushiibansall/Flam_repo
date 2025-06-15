import cv2
import numpy as np

# Load background image
image = cv2.imread("bg.jpg")
if image is None:
    print("Error: Couldn't load the background image.")
    exit()

# Convert to HSV (shadows are often low brightness but not low saturation)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define thresholds for "dark" regions (possible shadows)
lower_shadow = np.array([0, 0, 0])
upper_shadow = np.array([180, 255, 80])  # You can tweak the value threshold here

# Generate mask
shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

# Save mask
cv2.imwrite("shadow_mask.png", shadow_mask)
print("Shadow mask saved as shadow_mask.png")

# Optional: Visualize
cv2.imshow("Original", image)
cv2.imshow("Shadow Mask", shadow_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Laplacian to measure edge strength in the mask
laplacian = cv2.Laplacian(shadow_mask, cv2.CV_64F)
edge_strength = np.mean(np.abs(laplacian))

print(f"Edge Sharpness Score: {edge_strength:.2f}")
if edge_strength > 5:
    print("Detected: Hard Shadows")
else:
    print("Detected: Soft Shadows")
