import os
import cv2

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


class model:

    def __init__(self, file):
        self.file = file;

    def setFileNameScore(self, score):
        self.fileNameScore = score

    def setDistanceScore(self, score):
        self.distanceScore = score

    def setViewsScore(self, score):
        self.viewsScore = score

    def setTagsScore(self, score):
        self.tagsScore(self, score)

    # expects to get a path to a folder with pictures (devset/img)
    def faceDetect(self, folder):
        for filename in os.listdir(folder):
            if not os.path.isdir(filename):
                if ".jpg" in filename:
                    tmp = checkForFaces(folder + filename)
                    if tmp:
                        self.faceScore = 1
                    else:
                        self.faceScore = 0