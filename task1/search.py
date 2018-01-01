from xml.dom import minidom
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
import zipfile
import config
import cv2
import os
from model import *
import urllib.request
import numpy as np

class rate_method:
    images = []
    files = []
    name_score = []
    tag_score = []
    views_score = []
    distance_score = []
    face_score = []

    def rate(self, search):

        def rate_name(line):
            if search.encode().decode('utf-8').upper() in line.upper():
                self.name_score.append(1)
            else:
                self.name_score.append(0)

        # 1-Entfernung/(6371/2)
        def rate_distance(line):
            r = 6371  # Radius of earth in kilometers. Use 3956 for miles

            def haversine(lon2, lat2):
                # convert decimal degrees to radians
                geolocator = Nominatim()
                location = geolocator.geocode(search, timeout=None)
                lat1 = location.latitude
                lon1 = location.longitude
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                # haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                return c * r

            xml = minidom.parseString(line)
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('latitude') and photo.hasAttribute('longitude'):
                lat2 = float(photo.getAttribute('latitude'))
                lon2 = float(photo.getAttribute('longitude'))
                a = haversine(lon2, lat2)
                self.distance_score.append(1-(r/2)/a)
            else:
                self.distance_score.append(0.5) #TODO to discuss

        # 1-foundTags/numTags
        def rate_tags(line):
            unrelevant = ["karneval", "market", "riot", "protest", "demonstration"]  # TODO find more
            count = 0
            xml = minidom.parseString(line.encode('utf-8'))
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('tags'):
                tags = photo.getAttribute('tags')
                for u in unrelevant:
                    if u in tags:
                        count = count + 1
                        self.tag_score.append(1 - count / len(unrelevant))
                    else:
                        self.tag_score.append(0.5)

        def rate_face(line):
            def checkForFaces(url):
                resp = urllib.request.urlopen(url)
                image = np.array(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Only face in front of camera detected
                # with this parameters some pictures without faces are also detected
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03,
                                                      minNeighbors=25, minSize=(70, 70),
                                                      flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
                if len(faces) > 0:
                    return True
                else:
                    return False

            xml = minidom.parseString(line.encode('utf-8'))
            photo = xml.getElementsByTagName("photo")[0]
            url = photo.getAttribute('url_b')
            tmp = checkForFaces(url)
            if tmp:
                self.face_score.append(1)
            else:
                self.face_score.append(0)

        # 1 if in file name, else 0
        max_view = 0
        with zipfile.ZipFile(config.data_path+'desccred2.zip') as z:
            for filename in z.namelist():
                if not os.path.isdir(filename):
                    with z.open(filename, 'r') as f:
                        for line in f:
                            if "<photo ".encode() in line:
                                line = line.decode('utf-8')
                                self.files.append(line)
                                rate_name(line)
                                rate_distance(line)
                                rate_tags(line)
                                rate_face(line)

                                xml = minidom.parseString(line.encode('utf-8'))
                                photo = xml.getElementsByTagName("photo")[0]
                                if photo.hasAttribute('views'):
                                    views = int(photo.getAttribute('views'))
                                    if views > max_view:
                                        max_view = views
        #rate views
        for line in self.files:
            xml = minidom.parseString(line.encode('utf-8'))
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('views'):
                views = int(photo.getAttribute('views'))
                self.views_score.append(1-views/max_view)
            else:
                self.views_score.append(0.5)

        for i in range(0, len(s.files)):
            self.images.append(image(s.files[i], s.name_score[i], 0, s.views_score[i], s.tag_score[i], s.face_score[i]))

s = rate_method()
s.rate("Barcelona")
for i in s.images:
    print(i.file);
    scores = [i.name_score, i.tags_score, i.views_score, i.distance_score, i.face_score]
    print(scores)