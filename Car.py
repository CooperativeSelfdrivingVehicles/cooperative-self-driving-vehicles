# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:10:33 2019

@author: ElSaYeD
"""
from keras.models import load_model
import cv2
import numpy as np
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


model = load_model('model.h5')
speed_limit = 10



vs = cv2.VideoCapture(0)
(W, H) = (None, None)

count=0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
    (grabbed, image) = vs.read()
	
	# if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break
    	# of the stream
	

	# if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    
    
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    	
# read the next frame from the file
    if k==27:
        break
   

cv2.waitKey(0)
# release the file pointers
print("[INFO] cleaning up...")
vs.release()