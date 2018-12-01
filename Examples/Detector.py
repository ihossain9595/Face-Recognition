import cv2
import numpy as np

#Haarcascade library for face detection
faceDetector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Cam for video
cam = cv2.VideoCapture(0)

rec = cv2.face.createLBPHFaceRecognizer()
rec.load('recognizer/trainingData.yml')

#names = {"person name" : 1}
#name = names[id_]
Id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)

while(True) :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#For detecting the face
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

#For the rectangle around the face  
    for(x, y, w, h) in faces :
        color = (255, 0, 0)                             #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(img, (x,y), (end_cord_x, end_cord_y), color, stroke)

        roi_gray = gray[y:y+h, x:x+w]
        Id, conf = rec.predict(roi_gray)
        cv2.putText(img, str(Id), (x, y + h), fontFace, fontScale, fontColor)

#Displaying frame
    cv2.imshow('Face Recognition' ,img)

    

#Breaking key
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

#Release capture    
cam.release()
cv2.destroyAllWindows()