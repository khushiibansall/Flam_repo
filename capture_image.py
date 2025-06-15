import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("error")
    exit()

print("press SPACE")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed")
        break

    cv2.imshow("Press SPACE", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:
        break
    elif key % 256 == 32:
        img_name = "person_image.jpg"
        cv2.imwrite(img_name, frame)
        print(f"img saved as {img_name}")
        break

cap.release()
cv2.destroyAllWindows()

