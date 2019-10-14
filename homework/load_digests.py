from sklearn import metrics, cluster, datasets, mixture
from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

np.random.seed(42)
digits = load_digits()
data = scale(digits.data)
labels_true= digits.target

print(50 * '_')
print('Method name'.ljust(25) +  'NMI    homo   compl  ')

def bench(name, labels_true, labels_pred):
    print('%-25s%.3f\t%.3f\t%.3f'
          % (name,
             metrics.normalized_mutual_info_score(labels_true, labels_pred ,average_method='arithmetic'),
             metrics.homogeneity_score(labels_true, labels_pred),
             metrics.completeness_score(labels_true,labels_pred),
             ))


km = KMeans().fit(data)
labels_pred= km.labels_
bench('Kmeans',labels_true, labels_pred)

af = AffinityPropagation().fit(data)
labels_pred= af.labels_
bench('AffinityPropagation',labels_true, labels_pred)

ms = MeanShift(bandwidth=8).fit(data)
labels_pred= ms.labels_
bench('MeanShift',labels_true, labels_pred)

sc = SpectralClustering(n_clusters=8, affinity='nearest_neighbors').fit(data)
labels_pred= ms.labels_
bench('SpectralClustering',labels_true, labels_pred)

ag = AgglomerativeClustering().fit(data)
labels_pred= ag.labels_
bench('AgglomerativeClustering',labels_true, labels_pred)

db = DBSCAN(eps=4, min_samples=6).fit(data)
labels_pred= db.labels_
bench('DBSCAN',labels_true, labels_pred)

labels_pred = mixture.GaussianMixture(n_components=6).fit_predict(data)
bench('GaussianMixture',labels_true, labels_pred)
print(50 * '_')