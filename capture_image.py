# import cv2

# # Open the first webcam connected
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("Press SPACE to capture the image. Press ESC to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     cv2.imshow("Camera - Press SPACE to Capture", frame)

#     key = cv2.waitKey(1)
#     if key % 256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif key % 256 == 32:
#         # SPACE pressed
#         img_name = "person_image.jpg"
#         cv2.imwrite(img_name, frame)
#         print(f"Image saved as {img_name}")
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import os # Import the os module to check for file existence

# Specify the path to your input image file
# Make sure 'input_image.jpg' exists in the same directory as your script,
# or provide the full path, e.g., "C:/Users/YourUser/Pictures/my_photo.png"
input_image_path = "beach_person.jpg" # <--- IMPORTANT: Change this to your image file name

# Check if the image file exists
if not os.path.exists(input_image_path):
    print(f"Error: Image file not found at '{input_image_path}'.")
    print("Please make sure the image file exists and the path is correct.")
    exit()

# Read the image from the specified path
img = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Could not read the image from '{input_image_path}'.")
    print("This might happen if the file is corrupted or not a valid image format.")
    exit()

print(f"Displaying image: {input_image_path}. Press any key to close the window.")

# Display the image in a window
cv2.imshow("Input Image", img)

# Wait indefinitely until a key is pressed
# waitKey(0) means it waits forever for a key press
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()