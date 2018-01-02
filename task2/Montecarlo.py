import random as rand
import bitarray as ba
import clustering

'''
Searching parameters
'''
TRIES = 10

'''
Hyperparameters
- Features (any combination of the 10 features)
- Clustering Algorithm (preselected to Agglomerative, DBSCAN and Spectral)
- Settings for Clustering Algorithm:
  - Agglomerative
    - n_clusters (int > 1)
    - affinity ('euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed')
    - compute_full_tree fixed to True, because n_clusters will be much smaller than n_samples
    - linkage ('ward', 'complete', or 'average', attention: in case of 'ward' affinity is fixed to 'euclidean')
  - Spectral
    - n_clusters (int > 1)
    - eigen_solver (None, 'arpack', 'lobpcg', or 'amg', attention: 'amg' needs pyamg installed)
    - random_state fixed to None, because we assume that the random seed value will not influence the results that much
    - n_init (int) take a very close look on that!!!
    - gamma (float, attention: is ignored for affinity='nearest_neighbors')
    - affinity ('nearest_neighbors', 'precomputed', or 'rbf')
    - n_neighbors (int, attention: is ignored for affinity='rbf')
    - eigen_tol (float, only if eigen_solver='arpack')
    - assign_labels ('kmeans', or 'discretize')
  - DBSCAN
    - eps (float)
    - min_samples (int > 1)
    - algorithm ('auto', 'ball_tree', 'kd_tree', or 'brute')
    - p (float)
'''

print('Start montecarlo search with ' + str(TRIES) + ' tries...')

bestMin = {'value': 1.0, 'cnt': -1, 'params': None}
bestMean = {'value': 0.0, 'cnt': -1, 'params': None}
bestSD = {'value': 1.0, 'cnt': -1, 'params': None}
bestMedian = {'value': 0.0, 'cnt': -1, 'params': None}
bestMax = {'value': 0.0, 'cnt': -1, 'params': None}

cnts = [0, 0, 0]
first_loop = True

for cnt in range(TRIES):
    '''
    features_map
      BitArray that represents the chosen features
      From MSB to LSB these features are described:
        0 CM
        1 CM3x3
        2 CN
        3 CN3x3
        4 CSD
        5 GLRLM
        6 GLRLM3x3
        7 HOG
        8 LBP
        9 LBP3x3
      use random number with 10 Bit (unsigned int from 1 to 1023)
    '''
    features_map = ba.bitarray(('{0:010b}'.format(rand.randint(1, 1023)))[-10:])

    '''
    clustering_algorithm
      integer that represents the chosen clustering algorithm
      These 3 preselected algorithms are available:
        1 AgglomerativeClustering
        2 SpectralClustering
        3 DBSCAN
    '''
    clustering_algorithm = rand.randint(0, 2)

    '''
    n_clusters
      integer that represents the number of clusters to produce.
      This is only applied, in case of Agglomerative or Spectral clustering.

      Decisions for limits:
      1. Init: limit from 10 to 50
      2.
    '''
    n_clusters = rand.randint(10, 50)

    '''
    linkage
      string that describes which linkage criterion to use
    '''
    linkage_values = ['ward', 'complete', 'average']
    linkage = linkage_values[rand.randint(0, len(linkage_values)-1)]

    '''
    affinity
      string that defines the metric used to compute the linkage
      This is only applied in case of Agglomerative or Spectral clustering
      AglomerativeClustering values:
        'euclidean', 'l1', 'l2', 'manhattan', 'cosine', or 'precomputed'
      SpectralClustering values:
        'nearest_neighbors', 'precomputed', or 'rbf'
    '''
    affinity = None
    agglomerative_affinity_values = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    spectral_affinity_values = ['nearest_neighbors', 'precomputed', 'rbf']
    if clustering_algorithm == 0:
        # Agglomerative Clustering
        if linkage == 'ward':
            affinity = 'euclidean'
        else:
            affinity = agglomerative_affinity_values[rand.randint(0, len(agglomerative_affinity_values) - 1)]
    if clustering_algorithm == 1:
        # Spectral Clustering
        affinity = spectral_affinity_values[rand.randint(0, len(spectral_affinity_values) - 1)]

    '''
    compute_full_tree
      stops early construction of tree at n_cluster
      can be helpful regarding performance to set to False if n_cluster is in same dimension than n_samples
      => we fix it to True
    '''
    compute_full_tree = True

    '''
    eigen_solver
      string that describes the eigenvalue decomposition strategy
    '''
    eigen_solver_values = [None, 'arpack', 'lobpcg', 'amg']
    eigen_solver = eigen_solver_values[rand.randint(0, len(eigen_solver_values)-1)]

    '''
    random_state
      can be used to define a random number generator in some eigen_solver cases
      we do not use this => None
    '''
    random_state = None

    '''
    n_init
      int that defines the number of times a k-means algorithm will run with different seeds

      Decisions for limits:
      1. Init: limit from 5 to 20, because default is 10
      2.
    '''
    n_init = rand.randint(5, 20)

    '''
    gamma
      float that describes the kernel coefficient.
      is not used in case of 'nearest-neighbor' affinity

      Decisions for limits:
      1. Init: limit from 0.0 to 1.0, because no prior knowledge
      2.
    '''
    if affinity == 'nearest-neighbor':
        gamma = None
    else:
        gamma = rand.uniform(0.0, 1.0)

    '''
    n_neighbors
      int that describes number of neighbors to use when constructing the affinity matrix

      Decisions for limits:
      1. Init: limit from 3 to 50, because no prior knowledge
      2.
    '''
    if affinity == 'rbf':
        n_neighbors = None
    else:
        n_neighbors = rand.randint(3, 50)

    '''
    eigen_tol
      float that describes the stopping criterion for eigendecomposition

      Decisions for limits:
      1. Init: limit from 0.0 to 1.0, because no prior knowledge
      2.
    '''
    eigen_tol = rand.uniform(0.0, 1.0)

    '''
    assign_labels
      string that describes the strategy to use by assigning labels in the embedding space
    '''
    assign_labels_values = ['kmeans', 'discretize']
    assign_labels = assign_labels_values[rand.randint(0, len(assign_labels_values)-1)]

    '''
    eps
      float that describes the maximum distance between two values to be considered as in the same neighborhood

      Decisions for limits:
      1. Init: limit from 0.0 to 1.0, because no prior knowledge
      2.
    '''
    eps = rand.uniform(0.0, 1.0)

    '''
    min_samples
      int that defines the number of samples in a neighborhood for a point to be considered as a core point

      Decisions for limits:
      1. Init: limit from 3 to 50, because no prior knowledge
      2.
    '''
    min_samples = rand.randint(3, 50)

    '''
    algorithm
      string that describes the method used for calculating the distance between points
    '''
    algorithm_values = ['auto', 'ball_tree', 'kd_tree', 'brute']
    algorithm = algorithm_values[rand.randint(0, len(algorithm_values)-1)]

    '''
    p
      float that describes the power of the Minkowski metric

      Decisions for limits:
      1. Init: limit from 0.0 to 1.0, because no prior knowledge
      2.
    '''
    p = rand.uniform(0.0, 1.0)

    if first_loop:
        print("Min\tMean\tSD\tMedian\tMax")
        first_loop = False

    cnts[clustering_algorithm] += 1
    try:
        result = clustering.cluster(features_map=features_map, clustering_algorithm=clustering_algorithm, n_clusters=n_clusters,
                                    linkage=linkage, affinity=affinity, compute_full_tree=compute_full_tree,
                                    eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma,
                                    n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, eps=eps,
                                    min_samples=min_samples, algorithm=algorithm, p=p)
        print(str(result['min'])+"\t"+str(result['mean'])+"\t"+str(result['sd'])+"\t"+str(result['median'])+"\t"+str(result['max']))
    except Exception as ex:
        clustering_algorithm_values = ['Agglomerative', 'Spectral', 'DBSCAN']
        print("\n\nERROR:")
        print(type(ex))
        print(ex)
        print("\nParams")
        print("features_map="+str(features_map))
        print("clustering_algorithm="+str(clustering_algorithm_values[clustering_algorithm]))
        print("n_clusters="+str(n_clusters))
        print("linkage="+str(linkage))
        print("affinity="+str(affinity))
        print("compute_full_tree="+str(compute_full_tree))
        print("eigen_solver="+str(eigen_solver))
        print("random_state="+str(random_state))
        print("n_init="+str(n_init))
        print("gamma="+str(gamma))
        print("n_neighbors="+str(n_neighbors))
        print("eigen_tol="+str(eigen_tol))
        print("assign_labels="+str(assign_labels))
        print("eps="+str(eps))
        print("min_samples="+str(min_samples))
        print("algorithm="+str(algorithm))
        print("p="+str(p))
        print("\n\n")

print("\n\nAuswertung:")
print(str(cnts[0]) + " Agglomerative")
print(str(cnts[1]) + " Spectral")
print(str(cnts[2]) + " DBSCAN")