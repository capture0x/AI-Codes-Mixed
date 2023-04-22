import cv2
import numpy as np

img = cv2.imread("image.png")


img = cv2.resize(img, (600, 800))


color = cv2.bilateralFilter(img, 9, 250, 250)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)


edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)


quant = 9
div = 256 // quant
color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
color[:, :, 1] = div * (color[:, :, 1] // div)
color[:, :, 2] = div * (color[:, :, 2] // div)
color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)


cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("Original", img)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

