import bitarray as ba
from clustering import *
import time
from config import *

export_file_path = EXPORT_PATH + 'export_' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.csv'
print('Save results in: ' + export_file_path)

print('Start hyperparameter search...')

'''current best:'''
bestMin = {'value': 0.0}
bestMean = {'value': 0.0}
bestMedian = {'value': 0.0}
bestMax = {'value': 0.0}

def write_results(clustering_algorithm, n_clusters, results, first, features_map):
    with open(export_file_path, mode='a') as export_file:
        if first:
            export_file.write("Algorithm;n-clusters;Min;Mean;SD;Median;Max;Features\r")

        result_string = str(clustering_algorithm) + "\t" + str(n_clusters) + "\t" + str(results['min']) + "\t"\
                        + str(results['mean']) + "\t" + str(results['sd']) + "\t" + str(results['median']) + "\t"\
                        + str(results['max']) + "\t" + str(features_map) + "\r"
        result_string = result_string.replace('\t', ';')
        result_string = result_string.replace('.', ',')
        export_file.write(result_string)

        '''Check for new best:'''
        _min = ''
        _mean = ''
        _median = ''
        _max = ''
        if bestMin['value'] < results['min']:
            _min = '*'
            bestMin['value'] = results['min']
        if bestMean['value'] < results['mean']:
            _mean = '*'
            bestMean['value'] = results['mean']
        if bestMedian['value'] < results['median']:
            _median = '*'
            bestMedian['value'] = results['median']
        if bestMax['value'] < results['max']:
            _max = '*'
            bestMax['value'] = results['max']
        if _min is not '' or _mean is not '' or _median is not '' or _max is not '':
            print("\n\n####################")
            print(str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            print("New best result found for (n_clusters=" + str(n_clusters) + ", algorithm=" +
                  str(clustering_algorithm) + ", featuremap=" + str(featuremap) + ") with these values:")
            print("\tMin: " + str(bestMin['value']) + " " + _min)
            print("\tMean: " + str(bestMean['value']) + " " + _mean)
            print("\tSD: " + str(results['sd']))
            print("\tMedian: " + str(bestMedian['value']) + " " + _median)
            print("\tMax: " + str(bestMax['value']) + " " + _max)
            print("####################")

''' Search Main '''
algorithms = ['Agglomerative', 'Spectral', 'DBSCAN', 'Gaussian']
do_for_algos = [3]
first_run = True
# for given clustering algorithms
for algo in do_for_algos:
    print(str(time.strftime('\n\n%Y-%m-%d-%H-%M-%S', time.localtime())) + ": start with algorithm " + str(algorithms[algo]))
    # for 10, 15, 20, 25, and 30 clusters
    for nclust in range(10, 31, 5):
        if algo != 2:
            print(str(time.strftime('\n\n%Y-%m-%d-%H-%M-%S', time.localtime())) + ": start with " + str(nclust) + " clusters")
        # for every combination of 1 or more features
        for feature in range(1, 1024):
            try:
                if feature % 100 == 0:
                    print(str(time.strftime('\n\n%Y-%m-%d-%H-%M-%S', time.localtime())) + ": " + str(feature) + " features tested (" + str(feature*100/1024) + "%)")
                featuremap = ba.bitarray(('{0:010b}'.format(feature))[-10:])
                result = clustering.cluster(featuremap, algo, nclust)
                write_results(algorithms[algo], nclust, result, first_run, featuremap)
                first_run = False
            except Exception as ex:
                print("\n\n")
                print("--------------------")
                print(str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
                print("ERROR:")
                print(type(ex))
                print(ex)
                print("For search:")
                if algo != 2:
                    print("\tn_clusters = " + str(nclust))
                print("\talgorithm  = " + str(algorithms[algo]))
                print("\tfeaturemap = " + str(featuremap))
                print("--------------------")
        if algo == 2:
            break

print("\n\nSearch finished!")
