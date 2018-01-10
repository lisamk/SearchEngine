import random as rand
import bitarray as ba
import clustering
import time

'''
Searching parameters
'''
TRIES = 100
INFO = 5
EXPORT_PATH = "E:\\temp\\MultimediaSAR\\montecarlo\\"

'''
Hyperparameters
- Features (any combination of the 10 features)
- Clustering Algorithm (preselected to Agglomerative, DBSCAN and Spectral)
- Settings for Clustering Algorithm:
  - Agglomerative
    - n_clusters (int > 1)
    - affinity ('euclidean', 'l1', 'l2', 'manhattan', or 'cosine')
    - compute_full_tree fixed to True, because n_clusters will be much smaller than n_samples
    - linkage ('ward', 'complete', or 'average', attention: in case of 'ward' affinity is fixed to 'euclidean')
  - Spectral
    - n_clusters (int > 1)
    - eigen_solver (None, 'arpack'). Decision: 'lobpcg', and 'amg' ignored, due to producing lots of unsolveable errors)
    - random_state fixed to None, because we assume that the random seed value will not influence the results that much
    - n_init (int) take a very close look on that!!!
    - gamma (float, attention: is ignored for affinity='nearest_neighbors')
    - affinity ('nearest_neighbors', or 'rbf')
    - n_neighbors (int, attention: is ignored for affinity='rbf')
    - eigen_tol (float, only if eigen_solver='arpack')
    - assign_labels ('kmeans', or 'discretize')
  - DBSCAN
    - eps (float)
    - min_samples (int > 1)
    - algorithm ('auto', 'ball_tree', 'kd_tree', or 'brute')
    - p (float)
'''

export_file_path = EXPORT_PATH + 'export_' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.csv'
print('Save results in: ' + export_file_path)

print('Start montecarlo search with ' + str(TRIES) + ' tries...')

'''current best:'''
bestMin = {'value': 0.850489610699785, 'run': -1}
bestMean = {'value': 0.9308947015783102, 'run': -1}
bestMedian = {'value': 0.9365433109179113, 'run': -1}
bestMax = {'value': 0.9491357397051348, 'run': -1}
'''
params:
Best of;Min;Mean;SD;Median;Max;Features;Clustering Algorithm;Nr Clusters;Linkage;Affinity;Eigen Solver;N Init;Gamma;Nr Neighbors;Eigen Tolerance;Assign Labels;Eps;Min Samples;Algorithm;P;Compute Full Tree;Random State
min;0,850489610699785;0,9208822182089967;0,0195491632859;0,9244285051156343;0,9410816125860374;bitarray('0011000101');Spectral;60;average;rbf;arpack;18;0,5548634691620746;None;0,44707333740498056;kmeans;0,31798954854813366;6;kd_tree;0,5146490209583798;True;None
mean,median;0,8488177692858849;0,9308947015783102;0,0187286552409;0,9365433109179113;0,9484239959328927;bitarray('0010100111');Spectral;58;average;nearest_neighbors;None;6;0,22112548670749432;7;0,8750060516023664;discretize;0,4759734903129763;6;auto;0,8283175191480721;True;None
max;0,8435634105564843;0,9290948360681516;0,0195567939881;0,9361478969838501;0,9491357397051348;bitarray('0001000100');Spectral;59;complete;rbf;None;7;0,30669630330661446;None;0,3987867219650971;kmeans;0,9833346734769856;6;ball_tree;0,8058424195268572;True;None
'''

first_loop = True

def write_results(run, results, first, features_map, clustering_algorithm, n_clusters, linkage, affinity, eigen_solver,
                  n_init, gamma, n_neighbors, eigen_tol, assign_labels, eps, min_samples, algorithm, p,
                  compute_full_tree = True, random_state = None):
    with open(export_file_path, mode='a') as export_file:
        if first:
            export_file.write("Run;Min;Mean;SD;Median;Max;Features;Clustering Algorithm;Nr Clusters;Linkage;"
                              + "Affinity;Eigen Solver;N Init;Gamma;Nr Neighbors;Eigen Tolerance;Assign Labels;Eps;"
                              + "Min Samples;Algorithm;P;Compute Full Tree;Random State\r\n")

        result_string = str(run) + "\t" + str(results['min']) + "\t" + str(results['mean']) + "\t" + str(results['sd'])\
                        + "\t" + str(results['median']) + "\t" + str(results['max']) + "\t" + str(features_map) + "\t"\
                        + str(clustering_algorithm) + "\t" + str(n_clusters) + "\t" + str(linkage) + "\t"\
                        + str(affinity) + "\t" + str(eigen_solver) + "\t" + str(n_init) + "\t" + str(gamma) + "\t"\
                        + str(n_neighbors) + "\t" + str(eigen_tol) + "\t" + str(assign_labels) + "\t" + str(eps) + "\t"\
                        + str(min_samples) + "\t" + str(algorithm) + "\t" + str(p) + "\t" + str(compute_full_tree)\
                        + "\t" + str(random_state) + "\r\n"
        result_string = result_string.replace('\t', ';')
        result_string = result_string.replace('.', ',')
        export_file.write(result_string)

        '''Check for new best:'''
        _min = False
        if bestMin['value'] < results['min']:
            _min = True
            bestMin['value'] = results['min']
            bestMin['run'] = run
        _mean = False
        if bestMean['value'] < results['mean']:
            _mean = True
            bestMean['value'] = results['mean']
            bestMean['run'] = run
        _median = False
        if bestMedian['value'] < results['median']:
            _median = True
            bestMedian['value'] = results['median']
            bestMedian['run'] = run
        _max = False
        if bestMax['value'] < results['max']:
            _max = True
            bestMax['value'] = results['max']
            bestMax['run'] = run
        if _min or _mean or _median or _max:
            print("\n\n####################")
            print("New best result found for run <" + str(run) + "> with these values:")
            if _min:
                print("Min: " + str(bestMin['value']))
            if _mean:
                print("Mean: " + str(bestMean['value']))
            if _median:
                print("Median: " + str(bestMedian['value']))
            if _max:
                print("Max: " + str(bestMax['value']))
            print("####################\n\n")


''' MAIN '''
for run in range(TRIES):

    if (not first_loop) and ((run % INFO) == 0):
        print(str(run) + " tries finished (" + str((run/TRIES*100)) + "%)")

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
      
      Decisions:
        1. try only Spectral Clustering because of very good performances in the first tests
    '''
    clustering_algorithm_values = ['Agglomerative', 'Spectral', 'DBSCAN']
    #clustering_algorithm = rand.randint(0, len(clustering_algorithm_values)-1)
    clustering_algorithm = 1

    '''
    n_clusters
      integer that represents the number of clusters to produce.
      This is only applied, in case of Agglomerative or Spectral clustering.

      Decisions for limits:
      1. Init: limit from 10 to 50
      2. limit from 35 to 60 because of very good performances on spectral clustering
      3. limit from 53 to 60 because of best results on spectral clustering
    '''
    n_clusters = rand.randint(53, 60)

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
        'euclidean', 'l1', 'l2', 'manhattan', or 'cosine'
      SpectralClustering values:
        'nearest_neighbors', or 'rbf'
    '''
    affinity = None
    agglomerative_affinity_values = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    spectral_affinity_values = ['nearest_neighbors', 'rbf']
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
    eigen_solver_values = [None, 'arpack']#, 'lobpcg', 'amg']
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
      2. limit from 3 to 30, because of better performances on spectral clustering
      3. limit from 1 to 30, I want to try this value below 3
    '''
    if affinity == 'rbf':
        n_neighbors = None
    else:
        n_neighbors = rand.randint(1, 30)

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

    try:
        result = clustering.cluster(features_map=features_map, clustering_algorithm=clustering_algorithm, n_clusters=n_clusters,
                                    linkage=linkage, affinity=affinity, compute_full_tree=compute_full_tree,
                                    eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma,
                                    n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, eps=eps,
                                    min_samples=min_samples, algorithm=algorithm, p=p)

        write_results(run=run, results=result, first=first_loop,
                      features_map=features_map, clustering_algorithm=clustering_algorithm_values[clustering_algorithm], n_clusters=n_clusters,
                      linkage=linkage, affinity=affinity, compute_full_tree=compute_full_tree,
                      eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma,
                      n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, eps=eps,
                      min_samples=min_samples, algorithm=algorithm, p=p)
        first_loop = False
    except Exception as ex:
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

print("Montecarlo search finished!\n\n\n")