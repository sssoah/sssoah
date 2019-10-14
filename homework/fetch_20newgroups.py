import numpy as np
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import mixture

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]

print(50 * '_')
print('Method name'.ljust(25) +  'NMI    homo   compl  ')

def bench(name, labels_true, labels_pred):
    print('%-25s%.3f\t%.3f\t%.3f'
          % (name,
             metrics.normalized_mutual_info_score(labels_true, labels_pred ,average_method='arithmetic'),
             metrics.homogeneity_score(labels_true, labels_pred),
             metrics.completeness_score(labels_true,labels_pred),
             ))

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)

svd = TruncatedSVD(2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)


km = KMeans().fit(X)
labels_pred= km.labels_
bench('Kmeans',labels, labels_pred)

af = AffinityPropagation().fit(X)
labels_pred= af.labels_
bench('AffinityPropagation',labels, labels_pred)

ms = MeanShift().fit(X)
labels_pred= ms.labels_
bench('MeanShift',labels, labels_pred)

sc = SpectralClustering(n_clusters=true_k).fit(X)
labels_pred= ms.labels_
bench('SpectralClustering',labels, labels_pred)

ag = AgglomerativeClustering().fit(X)
labels_pred= ag.labels_
bench('AgglomerativeClustering',labels, labels_pred)

db = DBSCAN(eps=0.004, min_samples=6).fit(X)
labels_pred= db.labels_
bench('DBSCAN',labels, labels_pred)

labels_pred = mixture.GaussianMixture(n_components=6).fit_predict(X)
bench('GaussianMixture',labels, labels_pred)
print(50 * '_')