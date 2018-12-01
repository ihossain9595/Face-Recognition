import cv2
import numpy as np

#Haarcascade library for face detection
faceDetector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Cam for video
cam = cv2.VideoCapture(0)

#Username input
Id = input("Enter Username :")

sampleNum = 0

while(True) :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#For detecting the face
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces :

#For capturing image for dataset
        sampleNum = sampleNum + 1
        roi_gray = gray[y:y+h, x:x+w]                   #Region of interest
        img_item = "Dataset/User." + str(Id) + "." + str(sampleNum) + ".jpg"
        cv2.imwrite(img_item, roi_gray)

#For the rectangle around the face        
        color = (255, 0, 0)                             #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(img, (x,y), (end_cord_x, end_cord_y), color, stroke)
        cv2.waitKey(100)

#Displaying frame
    cv2.imshow('Face Recognition' ,img)

#Breaking key
    cv2.waitKey(1)
    if(sampleNum >=50) :
        break

#Release capture    
cam.release()
cv2.destroyAllWindows()