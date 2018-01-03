import pickle
from sklearn.cluster import *

AGGLOMERATIVE_CLUSTERER = 'agglomerative'
DBSCAN_CLUSTERER = 'dbscan'
SPECTRAL_CLUSTERER = 'spectral'
CLUSTERING_ALGORITHM = 'clustreringAlgorithm'


class Clusterer:
    """
    Clusterer is a class which handles all clustering things
    """

    '''
    creates an Agglomerative Clusterer
    '''
    @staticmethod
    def createAgglomerativeModel(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward'):
        return AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage)

    '''
    creates an DBSCAN Clusterer
    '''
    @staticmethod
    def createDBSCANModel(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=1):
        return DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)

    '''
    creates an Spectral Clusterer
    '''
    @staticmethod
    def createSpectralModel(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1):
        return SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, degree=degree, coef0=coef0, kernel_params=kernel_params, n_jobs=n_jobs)

    '''
    Returns the clustering object referenced by the name
    '''
    @staticmethod
    def getClusteringModel(name):
        return {AGGLOMERATIVE_CLUSTERER: Clusterer.createAgglomerativeModel(),
                DBSCAN_CLUSTERER: Clusterer.createDBSCANModel(),
                SPECTRAL_CLUSTERER: Clusterer.createSpectralModel()}[name]

    '''
    create empty Clusterer which have to be trained with data
    '''
    def __init__(self):
        self.model = None
        self.algorithm = None
        self.clusters = None

    '''
    creates Clusterer with saved parameters from previously trained Clusterer
    '''
    def __init__(self, paramsPath, clustersPath):
        with open(paramsPath, mode='r') as paramsFile:
            params = pickle.load(paramsFile)
        self.algorithm = params[CLUSTERING_ALGORITHM]
        self.model = Clusterer.getClusteringModel(self.algorithm)
        del params[CLUSTERING_ALGORITHM]
        self.model.set_params(params)

        with open(clustersPath, mode='r') as clustersFile:
            clusters = pickle.load(clustersFile)
        self.clusters = clusters

    '''
    Saves parameters of the Clusterer and the already clustered data into the given file.
    ATTENTION: The files are always overwritten!
    '''
    def saveClusterer(self, paramsPath, clustersPath):
        params = self.model.get_params(deep=True)
        params[CLUSTERING_ALGORITHM] = self.algorithm
        with open(paramsPath, mode='w') as paramsFile:
            pickle.dump(params, paramsFile)
        with open(clustersPath, mode='w') as clustersFile:
            pickle.dump(self.clusters, clustersFile)

    '''
    Sets the Clusterers model and algorithm
    '''
    def setModel(self, clusteringAlgorithm, clusteringModel):
        self.model = clusteringModel
        self.algorithm = clusteringAlgorithm

    '''
    Clusters the given data.
    Parameter 'features' has to contain the id in the first column and all other columns are feature columns
    '''
    def clusterData(self, dataset_name, features):
        if self.model is None:
            raise Exception("Create and set model before using it!!!")
        ids = features[0]
        features = features[1:]
        predictions = []
        self.model.fit(features, predictions)
        clusters = dict(zip(ids, predictions))
        self.clusters[dataset_name] = clusters

    '''
    Returns the cluster number for a given dataset and id (if this dataset was already clustered)
    '''
    def getCluster(self, dataset_name, sample_id):
        if self.clusters is None:
            return None
        if dataset_name not in self.clusters:
            return None
        dataset = self.clusters[dataset_name]
        if sample_id not in dataset:
            return None
        return dataset[sample_id]
