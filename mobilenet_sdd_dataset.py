# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:00:38 2024

@author: ASUS
"""

import numpy as np
import cv2


#preprocess
img_detected=cv2.imread('images/testing/scene5.jpg')
img_height=img_detected.shape[0]
img_width=img_detected.shape[1]
resized_image=cv2.resize(img_detected, (300,300))

#blob:-binary large object 
img_blob=cv2.dnn.blobFromImage(resized_image,0.007843,(300,300),127.5)


class_labels=['background','aeroplane','bicycle','boat','bus','bottle','bird','car','cat','cow','chair','dog','dining table','horse','motorbike','person','potted plant','sheep','sofa','train','tv/monitor']


mobilenetssd=cv2.dnn.readNetFromCaffe('datasets/mobilenetssd.prototext','datasets/mobilenetssd.caffemodel')
mobilenetssd.setInput(img_blob)
object_detection=mobilenetssd.forward()

no_of_detections=object_detection.shape[2]

for index in np.arange(0,no_of_detections):
    prediction_confidence=object_detection[0,0,index,2]
    
    if prediction_confidence > 0.4:
        predicted_class_index=int(object_detection[0,0,index,1])
        predicted_class_label=class_labels[predicted_class_index]
        
        #obtain the bounding box to the actual resized image
        bounding_box=object_detection[0,0,index,3:7] * np.array([img_width,img_height,img_width,img_height])
        (start_x_pt,start_y_pt,end_x_pt,end_y_pt)=bounding_box.astype("int")
        predicted_class_label="{}:{:.2f}%".format(class_labels[predicted_class_index], prediction_confidence*100)
        print("predicted object {}:{}".format(index+1,predicted_class_label))
        cv2.rectangle(img_detected, (start_x_pt,start_y_pt),(end_x_pt,end_y_pt),(0,255,0),2)
        cv2.putText(img_detected, predicted_class_label, (start_x_pt,start_y_pt-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0),1)
        
cv2.imshow("detected_objects",img_detected)
        