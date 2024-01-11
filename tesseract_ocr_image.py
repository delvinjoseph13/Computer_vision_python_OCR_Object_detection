# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#step 1 importing the libraries
from PIL import Image
import pytesseract
import cv2



#step2 load the image to ocr
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image_to_ocr=cv2.imread('images/testing/fox_sample1.png')


# step 3 prepocessing the image
#converting image to grey scale
preprocessed_img=cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)

#converting image to black and white image binary and otsu converting
preprocessed_img=cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#smooth the image using mediumblur
preprocessed_img=cv2.medianBlur(preprocessed_img, 3)


#step 4 load the the image to PIL image

cv2.imwrite('temp_img.png', preprocessed_img)
preprocessed_PIL_img=Image.open('temp_img.png')


#pass the PIL image to tesserat to ocr
text_extracted=pytesseract.image_to_string(preprocessed_PIL_img)
print(text_extracted)

#for showing the actual image
#cv2.imshow("Actual image",image_to_ocr)
