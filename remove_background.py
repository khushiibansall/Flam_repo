import cv2
from rembg import remove

# Load the captured image
input_path = "person_image.jpg"
output_path = "person_no_bg.png"

# Read image using OpenCV
input_image = cv2.imread(input_path)

# Convert image from BGR (OpenCV format) to RGB
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Remove background
output_image = remove(input_image_rgb)

# Save the result as PNG with transparency
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGRA))

print(f"Background removed and saved as {output_path}")
