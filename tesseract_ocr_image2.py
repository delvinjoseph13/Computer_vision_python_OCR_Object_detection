# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:11:48 2024

@author: ASUS
"""

from PIL import Image
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
image_to_ocr=cv2.imread("images/testing/fox_sample3.jpg")


preprocessed_img=cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)
preprocessed_img=cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
preprocessed_img=cv2.medianBlur(preprocessed_img, 3)



cv2.imwrite('temp_img2.jpg', preprocessed_img)
preprocessed_PIL_img=Image.open("temp_img2.jpg")

extract_word=pytesseract.image_to_string(preprocessed_PIL_img)
print(extract_word)

cv2.imshow("actual", image_to_ocr)