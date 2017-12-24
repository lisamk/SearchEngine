import os
import cv2

class faces:
    # Method for searching for faces
    def checkForFaces(file):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Only face in front of camera detected
        # with this parameters some pictures without faces are also detected
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.03,
                                              minNeighbors = 25, minSize=(70,70))
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if len(faces) > 0:
            return True
        else:
            return False


    # TODO: use the method for a ranking
    withFaces = []
    path = 'E:/Users/Gudrun/Documents/JKU/WS1718/Multimedia Search/div-2014/devset/img/img/aztec_ruins/'
    for filename in os.listdir(path):
        if not os.path.isdir(filename):
            if ".jpg" in filename:
                tmp = checkForFaces(path + filename)
                if tmp:
                    withFaces.append(filename)
    print(len(withFaces))