from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import statistics as st
import config

'''
This is a first start of task 2
The following steps are implemented

- One visual feature (CM feature) is read in from the folder of pre-computed features

- The real number of clusters for one location (Acropolis Athens) is read

- The image features of images that belong to athens are read

- The images are clustered by an agglomerative clustering algorithm

- the resulting images in the cluster are compared pair-wise with the ground truth


TODO:

- implement cross-validation
- experiment with the other features
    - perform feature selection

- perform calculations for all locations
- find a good average amount of clusters for the test data
- experiment with different clustering algorithms

'''

DATA_DIR = config.data_path
FEATURE_PATH = config.feature_path
GROUND_TRUTH_PATH = config.ground_truth_path

# do clustering for every location we have in the dev set
# read locations file
df_locations = pd.read_csv(DATA_DIR + "poiNameCorrespondences.txt", sep="\t", header=None)
# remove first column (names)
locations = np.array(df_locations[1])

score = []
for location in locations:
    # read in the file with the list of clusters in athens
    df_cluster = pd.read_csv(DATA_DIR + GROUND_TRUTH_PATH + location + " dclusterGT.txt", sep=",", header=None)
    # we only need the number of clusters for now
    n_clusters = len(df_cluster)

    # read the ground truth file for the images
    df_gt = pd.read_csv(DATA_DIR + GROUND_TRUTH_PATH + location + " dGT.txt", sep=",", header=None)
    # create a dictionary of the form { imageID : clusterID }
    truth = dict(zip(df_gt[0], df_gt[1]))

    # read in the image features
    df_feature = pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CM.csv", sep=",", header=None)
    # TODO add more features into the array here
    # read the ids into an array
    ids = np.array(df_feature[0])

    # remove the first column with the image ids
    df_feature = df_feature.drop([0], axis=1)
    # create an array of the feautures [f1,f2,....]
    features = np.array(df_feature)

    # use feature array and number of clusters from above
    # use DBSCAN because it does not need the number of clusters
    model = DBSCAN().fit(features)
    # create dictionary { imageID, predictedCluster }
    prediction = dict(zip(ids, model.labels_))

    # there isn't a ground truth for each image, so we can use the subset for comparision
    # additionally the predictions are now in the same order as the truth values
    prediction_subset = {x: prediction[x] for x in truth.keys() if x in prediction}

    # in the next step we pairwisely compare each key with each other
    # if they are in the same cluster in the each dictionary
    correct = 0
    wrong = 0
    for key1 in prediction_subset.keys():
        for key2 in prediction_subset.keys():
            if key1 != key2:
                if ((truth.get(key1) == truth.get(key2) and (
                            prediction_subset.get(key1) == prediction_subset.get(key2))) or (
                                truth.get(key1) != truth.get(key2) and (
                                    prediction_subset.get(key1) != prediction_subset.get(key2)))):
                    correct += 1
                else:
                    wrong += 1

    # the performance is measured by Rand index
    # https://en.wikipedia.org/wiki/Rand_index
    score.append(correct / (correct + wrong))

# calculate statistics over all scores
print("Min:    " + str(min(score)))
print("Mean:   " + str(sum(score) / len(score)))
print("Median: " + str(st.median(score)))
print("Max:    " + str(max(score)))