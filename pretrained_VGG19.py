# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:12:18 2024

@author: ASUS
"""

from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import cv2


img_path="images/testing/download.jpg"
img=load_img(img_path)

img=img.resize((224,224))

img_array=img_to_array(img)

img_array=np.expand_dims(img_array, axis=0)

img_array=imagenet_utils.preprocess_input(img_array)

predicted_model=VGG19(weights="imagenet")

predicted=predicted_model.predict(img_array)

actual_output=imagenet_utils.decode_predictions(predicted)
print(actual_output)

display_img=cv2.imread(img_path)
cv2.putText(display_img, actual_output[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (250,0,0))
cv2.imshow("predicted",display_img)