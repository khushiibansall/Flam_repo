import cv2
from rembg import remove
input_path = "person_image.jpg"
output_path = "person_no_bg.png"
input_image = cv2.imread(input_path)
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# remove bg
output_image = remove(input_image_rgb)
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGRA))

print(f"background removed and saved as {output_path}")
