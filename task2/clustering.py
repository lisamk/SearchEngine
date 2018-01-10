from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import numpy as np
import statistics as st
import bitarray as ba
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

def cluster(features_map, clustering_algorithm, n_clusters, linkage='ward', affinity=None, eigen_solver=None, n_init=10, gamma=1.0, n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', eps=0.5, min_samples=5, algorithm='auto', p=None, compute_full_tree = True, random_state = None):
    if affinity is None:
        if clustering_algorithm == 0:
            affinity = 'euclidean'
        if clustering_algorithm == 1:
            affinity = 'rbf'

    # do clustering for every location we have in the dev set
    # read locations file
    df_locations = pd.read_csv(DATA_DIR + "poiNameCorrespondences.txt", sep="\t", header=None)
    # remove first column (names)
    locations = np.array(df_locations[1])

    score = []
    firstLoop = True
    for location in locations:
        # read the ground truth file for the images
        df_gt = pd.read_csv(DATA_DIR + GROUND_TRUTH_PATH + location + " dGT.txt", sep=",", header=None)
        # create a dictionary of the form { imageID : clusterID }
        truth = dict(zip(df_gt[0], df_gt[1]))

        # read in the image features
        features_df = []
        # copy features map for every loop
        tmp_features_map = features_map
        if tmp_features_map >= ba.bitarray('1000000000'):
            # CM
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CM.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0111111111')
        if tmp_features_map >= ba.bitarray('0100000000'):
            # CM3x3
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CM3x3.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0011111111')
        if tmp_features_map >= ba.bitarray('0010000000'):
            # CN
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CN.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0001111111')
        if tmp_features_map >= ba.bitarray('0001000000'):
            # CN3x3
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CN3x3.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000111111')
        if tmp_features_map >= ba.bitarray('0000100000'):
            # CSD
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " CSD.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000011111')
        if tmp_features_map >= ba.bitarray('0000010000'):
            # GLRLM
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " GLRLM.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000001111')
        if tmp_features_map >= ba.bitarray('0000001000'):
            # GLRLM3x3
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " GLRLM3x3.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000000111')
        if tmp_features_map >= ba.bitarray('0000000100'):
            # HOG
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " HOG.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000000011')
        if tmp_features_map >= ba.bitarray('0000000010'):
            # LBP
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " LBP.csv", sep=",", header=None))
            tmp_features_map = tmp_features_map & ba.bitarray('0000000001')
        if tmp_features_map >= ba.bitarray('0000000001'):
            # LBP3x3
            features_df.append(pd.read_csv(DATA_DIR + FEATURE_PATH + location + " LBP3x3.csv", sep=",", header=None))

        # read the ids into an array
        ids = np.array(features_df[0][0])
        # start with id column and remove later
        #features = ids
        first = True
        for df_feature in features_df:
            # remove the first column with the image ids
            df_feature = df_feature.drop([0], axis=1)
            # create an array of all feautures [id,f1,f2,....]
            if first:
                features = np.array(df_feature)
            else:
                features = np.concatenate((features, np.array(df_feature)), axis=1)
            first = False
        # now remove the initially added ids
        #features = features[1:]

        # use feature array and number of clusters from above
        # use DBSCAN because it does not need the number of clusters
        if clustering_algorithm < 0:
            print("\n\nInvalid clustering algorithm: " + str(clustering_algorithm) + "!\n\n")
            return
        if clustering_algorithm == 0:
            model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, compute_full_tree=compute_full_tree, linkage=linkage)
        if clustering_algorithm == 1:
            model = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, n_jobs=1)
        if clustering_algorithm == 2:
            model = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, p=p, n_jobs=1)
        if clustering_algorithm > 2:
            print("\n\nInvalid clustering algorithm: " + str(clustering_algorithm) + "!\n\n")
            return
        model.fit(features)
        # create dictionary { imageID, predictedCluster }
        prediction = dict(zip(ids, model.labels_))

        # there isn't a ground truth for each image, so we can use the subset for comparision
        # additionally the predictions are now in the same order as the truth values
        prediction_subset = {x: prediction[x] for x in truth.keys() if x in prediction}

        # calculate performance using adjusted rand score:
        ars = adjusted_rand_score(list(truth.values()), list(prediction_subset.values()))
        # move score from [-1;1] to [0;1] and add to score array
        score.append(ars / 2 + 0.5)

    # calculate statistics over all scores
    return {'min': min(score), 'mean': (sum(score)/len(score)), 'sd': np.std(score), 'median': st.median(score), 'max': max(score)}

# Testing code
'''
algo = 1
n_clusters = 25
featuremap = ba.bitarray(('{0:010b}'.format(1))[-10:])
score = cluster(featuremap, algo, n_clusters)
print("Score for (n_clusters=" + str(n_clusters) + ", algorithm=" +
      str(algo) + ", featuremap=" + str(featuremap) + ") with these values:")
print("\tMin: " + str(score['min']))
print("\tMean: " + str(score['mean']))
print("\tSD: " + str(score['sd']))
print("\tMedian: " + str(score['median']))
print("\tMax: " + str(score['max']))
'''
