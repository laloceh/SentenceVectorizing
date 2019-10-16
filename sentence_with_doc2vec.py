#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:32:07 2018

@author: eduardo
"""

from sklearn import metrics

import gensim.models as g
import codecs
from sklearn import cluster
from nltk import word_tokenize

model = 'enwiki_dbow/doc2vec.bin'

#inference hyper-parameters
start_alpha = 0.01
infer_epoch = 1000

# load model
m = g.Doc2Vec.load(model)

sentences = [['this is the one good machine learning book'],
             ['this is another book'],
             ['one more book'],
             ['weather rain snow'],
             ['yesterday weather snow'],
             ['forecast tomorrow rain snow'],
             ['this is the new post'],
             ['this is about more machine leaning post'],
             ['and this is the one last post book']]
             
test_docs = [word_tokenize(sent[0]) for sent in sentences]               
print(test_docs)

X = []
for d in test_docs:
    X.append(m.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
    
k = 2
NUM_CLUSTERS = 2
kmeans = cluster.KMeans(n_clusters = NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print('Cluster id labels for inputed data', list(labels))

print('Kmeans score:', kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
print('Silhouette score:', silhouette_score)