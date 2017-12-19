import sklearn
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import scipy as sp

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

DATA_DIR = "D:/DATA/div-2014/devset/"
FEATURE_PATH = "descvis/img/"
GROUND_TRUTH_PATH = "gt/dGT/"

# read in the file with the list of clusters in athens
df_cluster = pd.read_csv(DATA_DIR + GROUND_TRUTH_PATH + "acropolis_athens dclusterGT.txt", sep=",", header=None)
# we only need the number of clusters for now
n_clusters = len(df_cluster)

# read the ground truth file for the images
df_gt = pd.read_csv(DATA_DIR + GROUND_TRUTH_PATH + "acropolis_athens dGT.txt", sep=",", header=None)
# create a dictionary of the form { imageID : clusterID }
truth = dict(zip(df_gt[0], df_gt[1]))

# read in the image features
df_feature = pd.read_csv(DATA_DIR + FEATURE_PATH + "acropolis_athens CM.csv", sep=",", header=None)
# read the ids into an array
ids = np.array(df_feature[0])

# remove the first column with the image ids
df_feature = df_feature.drop([0], axis=1)
# create an array of the feautures [f1,f2,....]
features = np.array(df_feature)

# use feature array and number of clusters from above
model = AgglomerativeClustering(n_clusters=n_clusters,
                                linkage="ward").fit(features)
# create dictionary { imageID, predictedCluster }
prediction = dict(zip(ids, model.labels_))

# there isn't a ground truth for each image, so we can use the subset for comparision
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
score = correct / (correct + wrong)
print(score)
