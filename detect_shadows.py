import cv2
import numpy as np

image = cv2.imread("bg.jpg")
if image is None:
    print("error")
    exit()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_shadow = np.array([0, 0, 0])
upper_shadow = np.array([180, 255, 80])  

# generating mask!!!
shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
cv2.imwrite("shadow_mask.png", shadow_mask)
print("shadow mask saved as shadow_mask.png")


cv2.imshow("Original", image)
cv2.imshow("Shadow Mask", shadow_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# using Laplacian 
laplacian = cv2.Laplacian(shadow_mask, cv2.CV_64F)
edge_strength = np.mean(np.abs(laplacian))

print(f"edge sharpness: {edge_strength:.2f}")
if edge_strength > 5:
    print("detected: hard shadows")
else:
    print("detected: soft shadows")
