# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:54:45 2024

@author: ASUS
"""

import numpy as np
import cv2

webcam_video_stream=cv2.VideoCapture(0)

while True:
    ret,current_frame=webcam_video_stream.read()
    img_detected=current_frame
    img_height=img_detected.shape[0]
    img_width=img_detected.shape[1]
    resized_image=cv2.resize(img_detected,(300,300))
    
    img_blob=cv2.dnn.blobFromImage(resized_image,0.007843,(300,300),127.5)
    
    class_labels=['background','aeroplane','bicycle','boat','bus','bottle','bird','car','cat','cow','chair','dog','dining table','horse','motorbike','person','potted plant','sheep','sofa','train','tv/monitor']
    
    mobilenetssd=cv2.dnn.readNetFromCaffe('datasets/mobilenetssd.prototext','datasets/mobilenetssd.caffemodel')
    mobilenetssd.setInput(img_blob)
    object_detection=mobilenetssd.forward()
    
    no_object_detection=object_detection.shape[2]
    
    for index in np.arange(0,no_object_detection):
        prediction_confidence=object_detection[0,0,index,2]
        
        if prediction_confidence>0.5:
            prediction_index=int(object_detection[0,0,index,1])
            prediction_label=class_labels[prediction_index]
            
            border_box=object_detection[0,0,index,3:7]*np.array([img_width,img_height,img_width,img_height])
            (start_x_pt,start_y_pt,end_x_pt,end_y_pt)=border_box.astype("int")
            prediction_class_label="{}:{:.2f}%".format(class_labels[prediction_index], prediction_confidence*100)
            print("predected object{}:{}".format(index+1, prediction_class_label))
            cv2.rectangle(img_detected, (start_x_pt,start_y_pt),(end_x_pt,end_y_pt), (0,255,0),2)
            cv2.putText(img_detected, prediction_class_label, (start_x_pt,start_y_pt-5), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0,255,0),1)
            

    cv2.imshow("predicted output", img_detected)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
webcam_video_stream.release()
cv2.destroyAllWindows()