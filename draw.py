import cv2
import numpy as np


img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_inv = 255 - gray
gray_blur = cv2.GaussianBlur(gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
output = cv2.divide(gray, 255 - gray_blur, scale=256)
cv2.imwrite('output_image.jpg', output)
