import cv2
import os

class faces:    # Method for searching for faces

    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

    path = "E:/Users/Gudrun/Documents/JKU/WS1718/Multimedia Search/div-2014/testset/img/"

    for root, dirs, files in os.walk(path, topdown=False):

        for name in dirs:
            #print(os.path.join(root, name))
            picpath = path + name
            rankFile = open('../faceRanksTestSetIMG/' + name + '.txt', 'w')
            for file in os.listdir(picpath):
                pic = picpath + '/'+ file
                id = os.path.splitext(file)[0]
                rankFile.write(str(id) + ',')
                img = cv2.imread(pic)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Only face in front of camera detected
                #  with this parameters some pictures without faces are also detected
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=25, minSize=(70, 70))
                if len(faces) > 0:
                    rankFile.write(str(1) + '\n')
                    print 1
                else:
                    rankFile.write(str(0) + '\n')
                    print 0
            rankFile.close()
