import numpy as np
import cv2 as cv

img = cv.imread('./test.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
cv.imshow('Original image',img)
cv.imshow('Gray image', gray)
  
cv.waitKey(0)
cv.destroyAllWindows()
