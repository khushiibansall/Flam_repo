import cv2
import numpy as np

image = cv2.imread("bg.jpg")
if image is None:
    print("Error: couldn't load bg.jpg")
    exit()

object_base = (200, 400)    
shadow_tip = (300, 500)    

clone = image.copy()
cv2.circle(clone, object_base, 5, (0, 255, 0), -1)
cv2.circle(clone, shadow_tip, 5, (0, 0, 255), -1)
cv2.line(clone, object_base, shadow_tip, (255, 0, 0), 2)

cv2.imwrite("light_direction_visualized.jpg", clone)
print("Saved annotated image as light_direction_visualized.jpg")

base = np.array(object_base)
tip = np.array(shadow_tip)
light_vector = base - tip

angle = np.degrees(np.arctan2(-light_vector[1], light_vector[0]))
print("\nEstimated Light Direction Vector (2D):", light_vector)
print(f"Estimated Light Angle (from horizontal): {angle:.2f}°")
