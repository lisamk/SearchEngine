import io
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import mean_squared_error

from config import *
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from task2 import *

import pickle
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
        faces_path = "faceScoresDev.csv"
    else:
        path = TEST_PATH
        topic = TEST_TOPIC_PATH
        text_path = TEST_TEXT_PATH
        text_poi_path = TEST_TEXT_POI_PATH
        faces_path = "faceScores.csv"

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
        views.append(np.log10(float(t.attrib["views"]) + 0.00001))
        ranks.append(float(t.attrib["rank"]))


        long_photo = float(t.attrib["longitude"])
        lat_photo = float(t.attrib["latitude"])

        if long_photo > 0 and lat_photo > 0:
            distances.append(haversine(long_poi, lat_poi, long_photo, lat_photo))
        else:
            distances.append(0.0)

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

    f = open(faces_path, "r")

    for line in f.readlines():
        id = int(line.split(",")[0].strip())
        face = float(line.split(",")[1][0])
        if id in features.keys():
            if len(features[id]) < 5:
               features[id].append(face)


    return features


def getRelevanceGt(location, dev=False):
    if dev:
        path = DEV_PATH + RELEVANCE_GROUND_TRUTH_PATH
    else:
        path = TEST_PATH + RELEVANCE_GROUND_TRUTH_PATH

    f = open(path + location + " rGT.txt", "r")
    truth = dict()

    for line in f.readlines():
        id = line.split(",")[0]
        r = line.split(",")[1]
        truth[int(id)] = int(r)

    return truth


def trainModelSingle(location, dev=False):
    big_dict = dict()
    big_truth_dict = dict()

    big_dict.update(getFeatures(location, dev=True))
    big_truth_dict.update(getRelevanceGt(location, dev=True))

    X = []
    Y = []
    for k in big_dict.keys():
        X.append(big_dict[k])
        Y.append(big_truth_dict[k])

    X_train = X
    Y_train = Y
    rf = RandomForestRegressor(n_estimators=10000, oob_score=True, random_state=1356128)
    rf.fit(X_train, Y_train)

    return rf


def trainModel(dev=False):
    big_dict = dict()
    big_truth_dict = dict()

    # big_dict.update(getFeatures("acropolis_athens", dev=True))
    # big_truth_dict.update(getGt("acropolis_athens", dev=True))

    for loc in getLocationNames(dev):
        big_dict.update(getFeatures(loc, dev))
        big_truth_dict.update(getRelevanceGt(loc, dev))

    X = []
    Y = []
    for k in big_dict.keys():
        X.append(big_dict[k])
        Y.append(big_truth_dict[k])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1356128)

    #    rf = RandomForestClassifier(n_estimators=10000, oob_score=True, random_state=1356128)
    rf = RandomForestRegressor(n_estimators=10000, oob_score=True, random_state=1356128)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)

    accuracy = accuracy_score(Y_test, (Y_pred > 0.5).astype(int))
    # pickle.dump(rf, open("trained_model", 'wb'))
    # joblib.dump(rf, "trained_data.pkl", protocol=2, compress=True)
    print(accuracy)
    print(mean_squared_error(Y_test, Y_pred))
    return rf


def trainModelFull():
    big_dict = dict()
    big_truth_dict = dict()

    # big_dict.update(getFeatures("acropolis_athens", dev=True))
    # big_truth_dict.update(getGt("acropolis_athens", dev=True))

    for loc in getLocationNames(dev=True):
        big_dict.update(getFeatures(loc, dev=True))
        big_truth_dict.update(getRelevanceGt(loc, dev=True))

    X = []
    Y = []
    for k in big_dict.keys():
        X.append(big_dict[k])
        Y.append(big_truth_dict[k])

    X_train = X
    Y_train = Y
    # rf = RandomForestClassifier(n_estimators=10000, oob_score=True, random_state=1356128)
    rf = RandomForestRegressor(n_estimators=10000, oob_score=True, random_state=1356128)
    rf.fit(X_train, Y_train)

    # pickle.dump(rf, open("trained_model", 'wb'))
    # joblib.dump(rf, "trained_data.pkl",protocol=2,compress=True)
    return rf


def predict(location, model, dev=False):
    f = getFeatures(location, dev)
    ids = []
    features = []
    for p in f.keys():
        ids.append(p)
        features.append(f[p])
    pred = model.predict(features)
    pred_dict = dict()
    for i in range(len(ids)):
        pred_dict[ids[i]] = pred[i]
    return pred_dict


def createPredictions(model, dev=False):
    f = open("relevanceData.csv", "w")
    for location in getLocationNames(dev):
        f.write(location)
        pred = predict(location, model, dev)
        for k in pred.keys():
            f.write("," + str(k) + "\t" + str(pred[k]))
        f.write("\n")


def getPredictionData(location):
    f = open("relevanceData.csv", "r")
    for line in f.readlines():
        l = line.split("\n")[0].split(",")
        if l[0] == location:
            id = []
            cl = []
            for loc in l[1:]:
                p = loc.split("\t")
                id.append(int(p[0]))
                cl.append(p[1])
            return dict(zip(id, cl))


def getResults(location, dev=False):
    pred = getPredictionData(location)
    result = reorderImages(pred, location, dev)
    return result


# print(getFeatures(9069965246, "acropolis_athens", dev=True))
start_time = time.time()


def getNumberOfRelevant(images, location, dev=False):
    gt_dict = getRelevanceGt(location, dev)

    correct = 0
    for i in images:
        if gt_dict[i] == 0:
            pass
        else:
            correct += 1

    return correct


def getPrecisionAtK(k, location, dev=False):
    imgs = getResults(location, dev=True)
    n = getNumberOfRelevant(imgs[:k], location, dev=True)
    prec = 0.0
    prec = n / len(imgs[:k])
    return prec


def printPrecisionAtK(dev=False):
    k_arr = [5, 10, 20, 50]

    precs1 = []
    precs2 = []
    precs3 = []
    precs4 = []

    for loc in getLocationNames(dev):

        ks = []
        # print(loc)
        for k in k_arr:
            prec = getPrecisionAtK(k, loc, dev)
            # print("Prec@k : " + str(k) + " : " + str(prec))
            ks.append(prec)
        precs1.append(ks[0])
        precs2.append(ks[1])
        precs3.append(ks[2])
        precs4.append(ks[3])

    print("\nPrecision@k=5")
    print("Average: " + str(np.average(precs1)))
    print("Median: " + str(np.median(precs1)))
    print("Min: " + str(np.min(precs1)))
    print("Max: " + str(np.max(precs1)))

    print("\nPrecision@k=10")
    print("Average: " + str(np.average(precs2)))
    print("Median: " + str(np.median(precs2)))
    print("Min: " + str(np.min(precs2)))
    print("Max: " + str(np.max(precs2)))

    print("\nPrecision@k=20")
    print("Average: " + str(np.average(precs3)))
    print("Median: " + str(np.median(precs3)))
    print("Min: " + str(np.min(precs3)))
    print("Max: " + str(np.max(precs3)))

    print("\nPrecision@k=50")
    print("Average: " + str(np.average(precs4)))
    print("Median: " + str(np.median(precs4)))
    print("Min: " + str(np.min(precs4)))
    print("Max: " + str(np.max(precs4)))


def getMaxNumberOfClustersGt(location, dev=False):
    if dev:
        path = DEV_PATH + CLUSTER_GROUND_TRUTH_PATH
    else:
        path = TEST_PATH + CLUSTER_GROUND_TRUTH_PATH

    f = open(path + location + " dGT.txt", "r")
    truth = dict()

    for line in f.readlines():
        id = line.split(",")[0]
        r = line.split(",")[1]
        truth[int(id)] = int(r)

    n_clusters = len(set(truth.values()))
    return n_clusters


def getNumberOfClustersGt(images, location, dev=False):
    if dev:
        path = DEV_PATH + CLUSTER_GROUND_TRUTH_PATH
    else:
        path = TEST_PATH + CLUSTER_GROUND_TRUTH_PATH

    f = open(path + location + " dGT.txt", "r")
    truth = dict()

    for line in f.readlines():
        id = line.split(",")[0]
        r = line.split(",")[1]
        truth[int(id)] = int(r)

    cl = []
    for i in images:
        if i in truth:
            cl.append(truth[i])

    n_clusters = len(set(cl))
    return n_clusters


def getNumberOfClusters(images, location, dev=False):
    pred = getClusterData(location)

    cl = []
    for i in images:
        cl.append(pred[i])

    n_clusters = len(set(cl))
    return n_clusters


def getRecallAtK(k, location, dev=False):
    imgs = getResults(location, dev=True)
    # n = getNumberOfClusters(imgs[:k], location, dev)
    n = getNumberOfClustersGt(imgs[:k], location, dev)
    m = getMaxNumberOfClustersGt(location, dev)
    recall = 0.0
    recall = n / min(m, k)
    return recall


def printRecallAtK(dev=False):
    k_arr = [5, 10, 20, 50]

    recalls0 = []
    recalls1 = []
    recalls2 = []
    recalls3 = []

    for loc in getLocationNames(dev):

        ks = []
        for k in k_arr:
            recall = getRecallAtK(k, loc, dev)
            # print("Recall@k : " + str(k) + " : " + str(recall))
            ks.append(recall)
        recalls0.append(ks[0])
        recalls1.append(ks[1])
        recalls2.append(ks[2])
        recalls3.append(ks[3])

    print("\nRecall@k=5")
    print("Average: " + str(np.average(recalls0)))
    print("Median: " + str(np.median(recalls0)))
    print("Min: " + str(np.min(recalls0)))
    print("Max: " + str(np.max(recalls0)))

    print("\nRecall@k=10")
    print("Average: " + str(np.average(recalls1)))
    print("Median: " + str(np.median(recalls1)))
    print("Min: " + str(np.min(recalls1)))
    print("Max: " + str(np.max(recalls1)))

    print("\nRecall@k=20")
    print("Average: " + str(np.average(recalls2)))
    print("Median: " + str(np.median(recalls2)))
    print("Min: " + str(np.min(recalls2)))
    print("Max: " + str(np.max(recalls2)))

    print("\nRecall@k=50")
    print("Average: " + str(np.average(recalls2)))
    print("Median: " + str(np.median(recalls2)))
    print("Min: " + str(np.min(recalls2)))
    print("Max: " + str(np.max(recalls2)))


# print(trainModel(dev=True))
# print(predict("acropolis_athens", dev=True))

# model = trainModelSingle("acropolis_athens", dev=True)
# print(predict("acropolis_athens", model, dev=True))

#start_time = time.time()
#model = trainModelFull()
#createPredictions(model,dev=False)
#print(getResults("ajanta_caves"))

#printPrecisionAtK(dev=True)
#printRecallAtK(dev=True)
#elapsed_time = time.time() - start_time
#print("Time: " + str(elapsed_time))

# f = open("team4_results.csv", "w")
# qid = 31
# for loc in getLocationNames():
#     print(loc)
#     i = 0
#     for r in getResults(loc)[:50]:
#         f.write(str(qid) + "\t0\t" + str(r) + "\t" + str(i) + "\t" + str(50 - i) + "\tfinal\n")
#         i += 1
#     qid += 1