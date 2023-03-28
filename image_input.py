import cv2
import sys
import os
import random

image_directory_path = sys.argv[1]
input_images = []

for path in os.listdir(image_directory_path):
    full_path = os.path.join(image_directory_path, path)
    if os.path.isfile(full_path):
        age = int(path[:path.find('_')])
        input_images.append((full_path, age))

random.shuffle(input_images)
image_iterator = iter(input_images)
current_image = next(image_iterator, -1)

while True:
    if current_image == -1:
        break

    (image_path, age) = current_image
    img = cv2.imread(image_path)
    img = cv2.putText(img, str(age), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                      (255, 255, 255), 2, cv2.LINE_AA, False)

    cv2.imshow('image', img)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if cv2.waitKey(20) & 0xFF == ord('n'):
        current_image = next(image_iterator, -1)


cv2.waitKey(0)
cv2.destroyAllWindows()
