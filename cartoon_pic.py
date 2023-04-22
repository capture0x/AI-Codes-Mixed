import cv2
import numpy as np

img = cv2.imread("image.png")

# resize image for faster processing
img = cv2.resize(img, (600, 800))

# apply bilateral filter and median filter
color = cv2.bilateralFilter(img, 9, 250, 250)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)

# detect edges using adaptive thresholding
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

# reduce number of colors
quant = 9
div = 256 // quant
color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
color[:, :, 1] = div * (color[:, :, 1] // div)
color[:, :, 2] = div * (color[:, :, 2] // div)
color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)

# combine color and edges
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("Original", img)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

