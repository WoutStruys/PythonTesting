import cv2
import numpy as np

green = np.uint8([[[10, 10, 10]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(hsv_green)
