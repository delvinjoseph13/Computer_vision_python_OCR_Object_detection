# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:16:07 2024

@author: ASUS
"""

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import cv2

#loading the image to predict
img_path='images/testing/test6.jpg'
img=load_img(img_path)


#converting the image to 224 x224 size

img=img.resize((224,224))

#converting the image to array

img_array=img_to_array(img)


#convert the image to 4 dimentional

img_array=np.expand_dims(img_array, axis=0)

#preprocess the input image_array

img_array=imagenet_utils.preprocess_input(img_array)

pretrained_model=VGG16(weights="imagenet")

prediction=pretrained_model.predict(img_array)

actual_prediction=imagenet_utils.decode_predictions(prediction)

print(actual_prediction)

display_img=cv2.imread(img_path)
cv2.putText(display_img,actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0))
cv2.imshow("predicted image",display_img)