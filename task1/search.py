from xml.dom import minidom
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
import os
import zipfile

class searchMethod:
    geolocator = Nominatim()
    lat1 = 0
    lon1 = 0

    def searchFor(word):
        global geolocator, lat1, lon1
        location = geolocator.geocode("word")
        lat1 = location.latitude
        lon1 = location.longitude

        relevantFilesList = []
        relevantLines = []

        with zipfile.ZipFile('E:/Users/Gudrun/Documents/JKU/WS1718/Multimedia Search/div-2014/devset/desccred.zip') as z:
            for filename in z.namelist():
              if not os.path.isdir(filename):
                    with z.open(filename, 'r') as f:
                        for line in f:
                            if word in line:
                                if filename not in relevantFilesList:
                                    relevantFilesList.append(filename)
                                    relevantLines.append(line)


        for name in relevantFilesList[:]:
            f = zipfile.ZipFile('E:/Users/Gudrun/Documents/JKU/WS1718/Multimedia Search/div-2014/devset/desccred.zip')
            xmlfile = f.open(name)
            xmldoc = minidom.parse(xmlfile)
            idlist = xmldoc.getElementsByTagName('photo')

        return relevantLines

    def filterByCoordinates(relevant):

        def haversine(lon2, lat2):
            # convert decimal degrees to radians
            global lat1, lon1
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers. Use 3956 for miles
            return c * r

        relevantLines = []
        for line in relevant:
            'TODO get test point from xml'
            test_point = [{'lat': -7.79457, 'lng': 110.36563}]

            lat2 = test_point[0]['lat']
            lon2 = test_point[0]['lng']

            radius = 10.00  # in kilometer

            a = haversine(lon2, lat2)

         #   print('Distance (km) : ', a)
            if a <= radius:
                relevantLines.append(line)

        return relevantLines

    'TODO'
    def filterByTags(relevant):
        return relevant

    'TODO'
    def filterByFlickr(relevant):
        return relevant

     # Filter
    filteredByName = searchFor(b'sydney')
    filteredByLocation = filterByCoordinates(filteredByName)
    filteredByTags = filterByTags(filteredByLocation)
    filteredByFlickr = filterByFlickr(filteredByTags);


