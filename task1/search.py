from xml.dom import minidom
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
import os
import zipfile

class searchMethod:

    def searchFor(word):
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

    def filterByCoordinates(relevant, search):
        relevantLines = []

        def haversine(lon2, lat2):
            # convert decimal degrees to radians
            geolocator = Nominatim()
            location = geolocator.geocode(search)
            lat1 = location.latitude
            lon1 = location.longitude
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers. Use 3956 for miles
            return c * r

        for line in relevant:
            xml = minidom.parseString(line)
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('latitude') and photo.hasAttribute('longitude'):
                lat2 = float(photo.getAttribute('latitude'))
                lon2 = float(photo.getAttribute('longitude'))
                radius = 10.00  #TODO
                a = haversine(lon2, lat2)
                if a <= radius:
                    relevantLines.append(line)
            else:
                relevantLines.append(line)

        return relevantLines

    def filterByTags(relevant):
        relevantLines = []
        unrelevant = ["karneval","market"] #TODO find more

        for line in relevant:
            xml = minidom.parseString(line)
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('tags'):
                tags = photo.getAttribute('tags')
                relevant = True
                for u in unrelevant:
                    if u in tags:
                        unrelevant = False
                        break
                if relevant:
                    relevantLines.append(line)
        return relevantLines

    'TODO'
    def filterByFlickr(relevant):
        relevantLines = []
        for line in relevant:
            xml = minidom.parseString(line)
            photo = xml.getElementsByTagName("photo")[0]
            if photo.hasAttribute('views'):
                views = int(photo.getAttribute('views'))
                if views > 100:
                    relevantLines.append(line)
        return relevantLines

    #FILTER
    search = "norway"
    filteredByName = searchFor(b'search')
  #  filteredByName = ["""<photo date_taken="2010-09-04 19:14:52" id="5004570811" tags="panorama toronto montreal niagrafalls sydney geiranger" title="Geiranger, Norway" url_b="http://farm5.static.flickr.com/4106/5004570811_0a860b6b68_b.jpg" userid="89093444@N00" views="262" />""",
  #                    """<photo date_taken="2005-02-11 14:27:52" id="4614474" latitude="-33.862919" longitude="151.212601" tags="hotel honeymoon sydney australia" title="Sydney:  Hotel Intercontinental" url_b="http://farm1.static.flickr.com/5/4614474_793ccae3d0_b.jpg" userid="89504146@N00" views="139" />""",
  #                    """<photo date_taken="2005-08-23 09:24:06" id="36456603" tags="sky clouds town sydney australia hasselblad" title="Sydney" url_b="http://farm1.static.flickr.com/30/36456603_9de5c110ef_b.jpg" userid="89826095@N00" views="56" />"""]
    filteredByLocation = filterByCoordinates(filteredByName, search)
    filteredByTags = filterByTags(filteredByLocation)
    filteredByFlickr = filterByFlickr(filteredByTags);
    for s in filteredByFlickr:
        print("Photo with searchtag in title: ", s)

