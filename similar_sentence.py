#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:28:57 2018

@author: eduardo

http://ai.intelligentonlinetools.com/ml/text-clustering-word-embedding-machine-learning/

For a sentence this is just taking the average of the word vectors
"""

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
    
from gensim.models import Word2Vec
#from nltk.cluster import KMeansClusterer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk import word_tokenize
import numpy as np
from sklearn import cluster
from sklearn import metrics
import random


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        print('word:', w)
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass
    return np.asarray(sent_vec) / numw


# Training data
sentences = [['this is the one good machine learning book'],
             ['this is another book'],
             ['one more book'],
             ['weather rain snow'],
             ['yesterday weather snow'],
             ['forecast tomorrow rain snow'],
             ['this is the new post'],
             ['this is about more machine leaning post'],
             ['and this is the one last post book']]
             

sentences = [word_tokenize(sent[0]) for sent in sentences]    
model = Word2Vec(sentences, min_count=1)

X = []
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))
    
print('='* 50)
print(X)

print(model.wv.vocab.keys())
print(len(model.wv.vocab))
print(model[model.wv.vocab])
print(model.wv.similarity('post', 'book'))
print(model.wv.most_similar(positive='machine'))


NUM_CLUSTERS = 2
kmeans = cluster.KMeans(n_clusters = NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print('Cluster id labels for inputed data', list(labels))
print('Centroids data', centroids)

print('Kmeans score:', kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
print('Silhouette score:', silhouette_score)


## Find similar sentence
test_sentence = ['a publication about machine learning']

test = word_tokenize(test_sentence[0])  
X_test = sent_vectorizer(test, model)
print(X_test)

knn = NearestNeighbors(n_neighbors=1)
knn.fit(X)

print('For "%s"' % test_sentence[0])
similar_sent = knn.kneighbors(X_test, return_distance=False)
print('The most similar sentence is: "%s"' % ' '.join(sentences[similar_sent[0][0]]))










