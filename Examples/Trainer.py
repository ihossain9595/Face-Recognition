import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer(1, 10, 10, 10)
path = 'Dataset'

def getImagesWithName(path) :
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    #current_id = 1
    #name_ids = {}
    faces = []
    Ids = []

    for imagePath in imagePaths :
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split('.')[1])

#Creating new name id current_id ++        
        #if not name in name_ids :
           # name_ids[name] = current_id
           # current_id = current_id + 1
      #  id_ = name_ids[name]


        faces.append(faceNp)
        print (Id)
        Ids.append(Id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces


Ids, faces = getImagesWithName(path)              #Getting image and names from the dataset folder
recognizer.train( faces, np.array(Ids))           #Training faces and names
recognizer.save('recognizer/trainingData.yml')      #Saving the trained file

cv2.destroyAllWindows()
