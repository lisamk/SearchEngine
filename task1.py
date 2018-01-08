import io
from sklearn.metrics import adjusted_rand_score

from config import *
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from utility_functions import *

import os

# from xml.dom.minidom import parse
from xml.etree import ElementTree as ET
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def getFeatures(location, dev=False):
    if dev:
        path = DEV_PATH
        topic = DEV_TOPIC_PATH
        text_path = DEV_TEXT_PATH
        text_poi_path = DEV_TEXT_POI_PATH
    else:
        path = TEST_PATH
        topic = TEST_TOPIC_PATH
        text_path = TEST_TEXT_PATH
        text_poi_path = TEST_TEXT_POI_PATH

    ids = []
    distances = []
    ranks = []
    views = []
    tags = []

    long_poi = 0.0
    lat_poi = 0.0
    x = ET.parse(path + topic)
    root = x.getroot()
    for t in root.iter("topic"):
        loc = t.find("title").text
        if loc == location:
            long_poi = float(t.find("longitude").text)
            lat_poi = float(t.find("latitude").text)
            break

    x = ET.parse(path + XML_PATH + location + ".xml")
    root = x.getroot()

    for t in root.iter("photo"):
        ids.append(int(t.attrib["id"]))
        views.append(np.log10(float(t.attrib["views"])+0.00001))
        ranks.append(float(t.attrib["rank"]))
        distances.append(haversine(long_poi, lat_poi, float(t.attrib["longitude"]), float(t.attrib["latitude"])))

    ranks = [i / max(ranks) for i in ranks]
    features = dict(zip(ids, [list(a) for a in zip(distances, ranks, views)]))

    poi_score = 0.0
    # f = open(path+text_poi_path,"r")
    f = io.open(path + text_poi_path, mode="r", encoding="utf-8")
    for line in f.readlines():
        splt = line.split("\"")
        loc = splt[0].split(" ")[0]
        if loc == location:
            for i in range(1, len(splt) - 1, 2):
                tf = float(splt[i + 1].split(" ")[1])
                df = float(splt[i + 1].split(" ")[2])
                tf_idf = tf * np.log10(298 / df)
                poi_score += tf_idf
            break

    f = io.open(path + text_path, mode="r", encoding="utf-8")
    for line in f.readlines():
        splt = line.split("\"")
        im = int(splt[0].strip())

        if im in features.keys():
            score = 0.0
            for i in range(1, len(splt) - 1, 2):
                tf = float(splt[i + 1].split(" ")[1])
                df = float(splt[i + 1].split(" ")[2])
                tf_idf = tf * np.log10(298 / df)
                score += tf_idf

            features[im].append(score / poi_score)

    return features


# print(getFeatures(9069965246, "acropolis_athens", dev=True))
start_time = time.time()

# for im in getImagesDev("acropolis_athens"):
fs = getFeatures("acropolis_athens", dev=True)

big_dict = dict()
for loc in getLocationNames(dev=True):
    big_dict.update(getFeatures(loc, dev=True))

print(big_dict)

elapsed_time = time.time() - start_time
print(elapsed_time)
