from xml.dom import minidom
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
import os
import zipfile
import config
import model

class searchMethod:

    def rate(search):

        def rateFileName(word):
            relevantFilesList = []
            files = []


            # 1 if in file name, else 0
            with zipfile.ZipFile(config.data_path+'/desccred.zip') as z:
                for filename in z.namelist():
                  if not os.path.isdir(filename):
                        with z.open(filename, 'r') as f:
                            for line in f:
                                m = model(line)
                                files.append(m)
                                if filename not in relevantFilesList:
                                    relevantFilesList.append(filename)
                                    if word in line:
                                        m.setFileNameScore(1)
                                    else:
                                        m.setFileNameScore(0)
                                       # relevantLines.append(line)


            # for name in relevantFilesList[:]:
            #     f = zipfile.ZipFile(config.data_path+'/desccred.zip')
            #     xmlfile = f.open(name)
            #     xmldoc = minidom.parse(xmlfile)
            #     idlist = xmldoc.getElementsByTagName('photo')

            return files

        # 1-Entfernung/(6371/2)
        def rateByCoordinates(model, search):
            r = 6371  # Radius of earth in kilometers. Use 3956 for miles

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
                return c * r

            for m in model:
                xml = minidom.parseString(m.file)
                photo = xml.getElementsByTagName("photo")[0]
                if photo.hasAttribute('latitude') and photo.hasAttribute('longitude'):
                    lat2 = float(photo.getAttribute('latitude'))
                    lon2 = float(photo.getAttribute('longitude'))
                  #  radius = 10.00
                    a = haversine(lon2, lat2)
                    m.setDistanceScore(1-(r/2)/a)
                else:
                    m.setDistanceScore(0) #TODO

            return model

        # 1-foundTags/numTags
        def rateByTags(model):
            unrelevant = ["karneval","market"] #TODO find more

            for m in model:
                count = 0
                xml = minidom.parseString(m.file)
                photo = xml.getElementsByTagName("photo")[0]
                if photo.hasAttribute('tags'):
                    tags = photo.getAttribute('tags')
                    for u in unrelevant:
                        if u in tags:
                            count = count + 1
                    m.setTagsScore(1-count/len(unrelevant))
            return m

        # 1-views/maxviews
        def rateByViews(model):
            maxViews = 0
            for m in model:
                xml = minidom.parseString(m.file)
                photo = xml.getElementsByTagName("photo")[0]
                if photo.hasAttribute('views'):
                    views = int(photo.getAttribute('views'))
                    if views>maxViews:
                        maxViews = views
            for m in model:
                xml = minidom.parseString(m.file)
                photo = xml.getElementsByTagName("photo")[0]
                if photo.hasAttribute('views'):
                    views = int(photo.getAttribute('views'))
                    m.setViewsScore(1-views/maxViews)
            return model

        rated = rateFileName(search)
     #   rated = rateByCoordinates(rated, search)
     #   rated = rateByTags(rated)
     #   return rateByViews(rated);

    result = rate("norway")
    for s in result:
        print(s.file);
