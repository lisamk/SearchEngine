import random
import operator
from audioop import reverse
import time
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from config import *
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


def getLocationList(dev=False):
    if dev:
        path = DEV_PATH
    else:
        path = TEST_PATH

    f = open(path + "poiNameCorrespondences.txt", "r")
    text = f.read().split("\n")
    locations = []
    for line in text:
        l = line.split("\t")
        locations.append(list(reversed(l)))

    # f = open(dev_path + "poiNameCorrespondences.txt", "r")
    # text = f.read().split("\n")
    # for line in text:
    #     l = line.split("\t")
    #     locations.append(list(reversed(l)))

    locations = sorted(locations, key=lambda l: l[0])

    return locations


def getLocationNames(dev=False):
    if dev:
        path = DEV_PATH
    else:
        path = TEST_PATH

    f = open(path + "poiNameCorrespondences.txt", "r")
    text = f.read().split("\n")
    locations = []
    for line in text:
        l = line.split("\t")
        locations.append(l[1])
    return locations


# def getImgLinks(location_name):
#     f = open("imgFolderContent.csv", "r")
#     text = f.read().split("\n")
#     locations = []
#     for line in text:
#         l = line.split(",")
#         if l[0] == location_name:
#             return l[1:]


def getImages(location_name):
    f = open("imgFolderContent.csv", "r")
    text = f.read().split("\n")
    names = []
    for line in text:
        l = line.split(",")
        if l[0] == location_name:
            for loc in l[1:]:
                names.append(loc.split(".")[0])
            return names

def getImagesDev(location_name):
    f = open("imgFolderContent_Dev.csv", "r")
    text = f.read().split("\n")
    names = []
    for line in text:
        l = line.split(",")
        if l[0] == location_name:
            for loc in l[1:]:
                names.append(loc.split(".")[0])
            return names


def clusterLocation(location_name, dev=False):
    if dev:
        path = DEV_PATH
    else:
        path = TEST_PATH

    n_clusters = 25
    df_feature = pd.read_csv(path + FEATURE_PATH + location_name + " CM.csv", sep=",", header=None)
    ids = np.array(df_feature[0])
    df_feature = df_feature.drop([0], axis=1)
    features = np.array(df_feature)

    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage="ward").fit(features)
    prediction = dict(zip(ids, model.labels_))
    return prediction


def createClusterFile(dev=False):
    f = open("clusterData.csv", "w")
    for location in getLocationNames(dev):
        f.write(location)
        pred = clusterLocation(location, dev)
        for k in pred.keys():
            f.write("," + str(k) + "\t" + str(pred[k]))
        f.write("\n")


def getClusterData(location_name):
    f = open("clusterData.csv", "r")
    for line in f.readlines():
        l = line.split("\n")[0].split(",")
        if l[0] == location_name:
            id = []
            cl = []
            for loc in l[1:]:
                p = loc.split("\t")
                id.append(int(p[0]))
                cl.append(p[1])
            return dict(zip(id, cl))


def evaluateClustering():
    createClusterFile(dev=True)
    scores = []
    for location in getLocationNames(dev=True):
        df_gt = pd.read_csv(DEV_PATH + GROUND_TRUTH_PATH + location + " dGT.txt", sep=",", header=None)
        truth = dict(zip(df_gt[0], df_gt[1]))
        pred = getClusterData(location)

        truth_arr = []
        pred_arr = []
        for t in truth.keys():
            truth_arr.append(truth[t])
            pred_arr.append(pred[t])

        score = adjusted_rand_score(truth_arr, pred_arr)
        # print(location + ": " + str(score))
        scores.append(score)

    scores = np.array(scores)

    avg = np.sum(scores) / len(scores)
    median = np.median(scores)
    minimum = np.min(scores)
    maximum = np.max(scores)
    # print("\nEvaluated with adjusted rand index")
    # print("Avg: " + str(avg))
    # print("Median: " + str(median))
    # print("Min: " + str(min))
    # print("Max: " + str(max))

    return avg, median, minimum, maximum


'''
        0 CM            512
        1 CM3x3         256
        2 CN            128
        3 CN3x3         64
        4 CSD           32
        5 GLRLM         16
        6 GLRLM3x3      8
        7 HOG           4
        8 LBP           2
        9 LBP3x3        1
    '''


def parameterSearch():
    results = np.zeros((30, 1024, 1))
    location_array = getLocationNames(dev=True)
    print(location_array)
    feature_array = []

    f = open("result.csv", "w")

    for j in range(len(location_array)):
        location = location_array[j]
        df_feature = pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CM.csv", sep=",", header=None)
        df_feature = df_feature.drop([0], axis=1)

        # LBP3x3
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " LBP3x3.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # LBP
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " LBP.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # HOG
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " HOG.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # GLRLM3x3
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " GLRLM3x3.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # GLRLM
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " GLRLM.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # CSD
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CSD.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # CN3x3
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CN3x3.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # CN
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CN.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # CM3x3
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CM3x3.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)
        # CM
        df_feature = (pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CM.csv", sep=",", header=None))
        df_feature = df_feature.drop([0], axis=1)
        feature_array.append(df_feature)

        for i in range(1024):
            val = i + 1
            first = True
            features = []
            if val >= 512:
                val -= 512
                if first:
                    first = False
                    features = feature_array[int(np.log2(512))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(512))]), axis=1)
            if val >= 256:
                val -= 256
                if first:
                    first = False
                    features = feature_array[int(np.log2(256))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(256))]), axis=1)
            if val >= 128:
                val -= 128
                if first:
                    first = False
                    features = feature_array[int(np.log2(128))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(128))]), axis=1)
            if val >= 64:
                val -= 64
                if first:
                    first = False
                    features = feature_array[int(np.log2(64))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(64))]), axis=1)
            if val >= 32:
                val -= 32
                if first:
                    first = False
                    features = feature_array[int(np.log2(32))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(32))]), axis=1)
            if val >= 16:
                val -= 16
                if first:
                    first = False
                    features = feature_array[int(np.log2(16))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(16))]), axis=1)
            if val >= 8:
                val -= 8
                if first:
                    first = False
                    features = feature_array[int(np.log2(8))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(8))]), axis=1)
            if val >= 4:
                val -= 4
                if first:
                    first = False
                    features = feature_array[int(np.log2(4))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(4))]), axis=1)
            if val >= 2:
                val -= 2
                if first:
                    first = False
                    features = feature_array[int(np.log2(2))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(2))]), axis=1)
            if val >= 1:
                val -= 1
                if first:
                    first = False
                    features = feature_array[int(int(np.log2(1)))]
                else:
                    features = np.concatenate((features, feature_array[int(np.log2(1))]), axis=1)

            n_clusters = 25
            df_feature = pd.read_csv(DEV_PATH + FEATURE_PATH + location + " CM.csv", sep=",", header=None)
            ids = np.array(df_feature[0])
            model = AgglomerativeClustering(n_clusters=n_clusters,
                                            linkage="ward").fit(features)
            prediction = dict(zip(ids, model.labels_))
            df_gt = pd.read_csv(DEV_PATH + GROUND_TRUTH_PATH + location + " dGT.txt", sep=",", header=None)
            truth = dict(zip(df_gt[0], df_gt[1]))

            truth_arr = []
            pred_arr = []
            for t in truth.keys():
                if t in prediction.keys():
                    truth_arr.append(truth[t])
                    pred_arr.append(prediction[t])

            score = adjusted_rand_score(truth_arr, pred_arr)
            results[j, i] = score

        print("\nResult for " + location_array[j])
        print("Min: " + str(np.min(results[j,])))
        print("Max: " + str(np.max(results[j,])))
        print("Median:" + str(np.median(results[j,])))
        print("Average: " + str(np.average(results[j,])))

        print("Best: " + str(np.argmax(results)))
        f.write(location_array[j] + "," + str(np.min(results[j,])) + "," + str(np.max(results[j,])) + "," + str(
            np.median(results[j,])) + "," + str(np.average(results[j,])) + "\n")


def reorderImages(image_dict, location_name, dev=False):
    data = getClusterData(location_name)
    sorted_imgs = sorted(image_dict.items(), key=operator.itemgetter(1), reverse=True)

    ranked_list = []

    while len(ranked_list) < len(sorted_imgs):
        used_clusters = []

        for i in sorted_imgs:
            if data[int(i[0])] not in used_clusters:
                if i[0] not in ranked_list:
                    used_clusters.append(data[int(i[0])])
                    ranked_list.append(i[0])

    return ranked_list


# imgs = [
#     10045759763,
#     10045795353,
#     12045021564,
#     135115202,
#     135115501,
#     509939988,
#     509939994,
#     509940010,
#     2183292799,
#     4910005618,
#     3897478187,
#     3952588018,
#     4309636706,
#     4309531528,
#     4308895697,
#     5042415222,
#     5251535492,
#     135115983,
#     135116101,
#     2184089934,
#     2183301183,
#     2951069467,
#     3472330055,
#     2951944040,
#     3473162568,
#     4316702171,
#     2951917734,
#     5251534830,
#     135116324,
#     2183279459,
#     135116398,
#     2951064223,
#     2951061755,
#     2951910186,
#     4312639443,
#     4313364034,
#     4922244491,
#     4928814004,
#     4930867307,
#     4934528512,
#     533818818,
#     5937150857,
#     5950693927,
#     5999524850,
#     5998990995,
#     5998982367,
#     5999531210,
#     5999532820,
#     5999518796,
#     5998975793,
#     5998981575,
#     12044979074,
#     4898405013,
#     4904458042,
#     4906971826,
#     542075748,
#     5655937322,
#     5950655857,
#     135114844,
#     10045898593,
#     288051306,
#     509940004,
#     5251535876,
#     9069976672,
#     457722692,
#     461593568,
#     457718082,
#     457721934,
#     461597998,
#     512080155,
#     3898269458,
#     3633366466,
#     5041829373,
#     512080179,
#     5250932835,
#     9069968528,
#     9067746215,
#     9067744841,
#     9067745863,
#     8877216049,
#     10045378096,
#     10045488655,
#     10045633316,
#     10045606263,
#     12044649575,
#     288044415,
#     2950994233,
#     457715096,
#     5250931819,
#     3951806833,
#     4316746457,
#     4317482112,
#     4909428001,
#     5251535082,
#     8877830126,
#     8877220281,
#     10045482563,
#     10045632093,
#     12044924063,
#     4316743975,
#     4316751509,
#     4317474802,
#     470006382,
#     470019825,
#     5251535776,
#     5251534158,
#     544579844,
#     9069966916,
#     8492197487,
#     8493296918,
#     135116270,
#     3536116282,
#     4476079181,
#     4476079117,
#     4476079309,
#     4476079389,
#     4476855160,
#     4476855358,
#     4476856682,
#     4476856270,
#     4476856880,
#     4476856946,
#     10045665564,
#     10045809175,
#     132872354,
#     12045478526,
#     135115675,
#     135115317,
#     2951841806,
#     2951859592,
#     3952607184,
#     4316736733,
#     4316681859,
#     4317472438,
#     466220980,
#     5250933407,
#     462896945,
#     461602133,
#     3951789739,
#     5251533966,
#     9069973156,
#     9067742495,
#     8492309101,
#     5098166243,
#     2951861710,
#     512080165,
#     2951846628,
#     2811278670,
#     2951102363,
#     3632560595,
#     4316754111,
#     470032783,
#     512080161,
#     5251533776,
#     544696551,
#     8877804938,
#     8877205941,
#     8492075597,
#     6231951817,
#     10045579766,
#     10045665523,
#     10045868856,
#     4918999221,
#     5041767379,
#     5041772573,
#     5041872853,
#     5042020745,
#     544700743,
#     544706421,
#     8877813508,
#     3951834477,
#     3952579860,
#     3951800783,
#     8877217453,
#     8877221455,
#     8877204767,
#     8877203469,
#     4316759251,
#     4317495448,
#     4317467692,
#     461601946,
#     509940000,
#     10045737995,
#     8877200925,
#     3472230091,
#     461600955,
#     8493324986,
#     8493294360,
#     8492178225,
#     8877218909,
#     2183306153,
#     2183308589,
#     6798572025,
#     4309475372,
#     4317497880,
#     4912099789,
#     9069964412,
#     9067737103,
#     8877807358,
#     8493307976,
#     3473255564,
#     5250931111,
#     3474151758,
#     5251533840,
#     9069963392,
#     9067739127,
#     9067738777,
#     9067738157,
#     8493392970,
#     5985080896,
#     5984516769,
#     5984516153,
#     5937710790,
#     5937151975,
#     5999647578]
#
# img_dict = dict(zip(imgs, range(len(imgs))))




# print(clusterLocation("acropolis_athens",dev=False))
# print(createClusterFile(dev=False))
# print(getClusterData("acropolis_athens"))
# evaluateClustering()

# print(reorderImages(img_dict, "acropolis_athens", dev=True))

#start_time = time.time()
#parameterSearch()
#elapsed_time = time.time() - start_time
#print(elapsed_time)
