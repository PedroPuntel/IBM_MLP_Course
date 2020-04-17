# 31/12/2019
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# DBSCAN Example (Density Based Spatial Clusterig of Applications with Noise)

#%% Modules
import wget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

#%% Function that creates random data points
def createRandomDataPoints(centroidLocation, num_samples, clusterDeviation):
    X, y = make_blobs(n_samples=num_samples, centers=centroidLocation, cluster_std=clusterDeviation)
    X = StandardScaler().fit_transform(X)
    return X, y

#%% Generating random data points
X, y = createRandomDataPoints(centroidLocation=[[4,3], [2,-1], [-1,4]], num_samples=1500, clusterDeviation=0.5)

#%% Fitting the DBSCAN model usign (R=0.3, M=7)
radius = 0.3
minsamples = 7
db_fit = DBSCAN(eps=radius, min_samples=minsamples).fit(X)
fit_labels = db_fit.labels_

#%% Labeling outlier data points
core_dp = np.zeros_like(a=fit_labels, dtype=bool)
core_dp[db_fit.core_sample_indices_] = True

#%% True number of clusters in the dataset (ignores outliers)
n_clusters = len(set(fit_labels)) - (1 if -1 in fit_labels else 0)

#%% Unique cluster labels
fit_labels_unique = set(fit_labels)

#%% Visualizing the clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(fit_labels_unique)))

for k, col in zip(fit_labels_unique, colors):
    
    # If the data point is an outlier
    if k == -1:
        
        # than its color will be black
        col = "black"
    
    # cluster that the data point belongs to
    dp_cluster = (fit_labels == k)
    
    # Plots the data points that are clustered
    xy = X[dp_cluster & core_dp]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=col, marker=u'o')

    # Plots the outliers
    xy = X[dp_cluster & ~core_dp]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=col, marker=u'o')
    
    # Additional plot tweaks
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Distribution - DBSCAN Algorithm')

#%% Comparing K-Means clustering and DBSCAN of the same dataset
from sklearn.cluster import KMeans 
k_means = KMeans(init = "k-means++", n_clusters=3, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

for k, col in zip(k_means_labels, colors):
        
    # cluster that the data point belongs to
    dp_cluster = (k_means_labels == k)
    
    # Plots the data points that are clustered
    xy = X[dp_cluster]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=col, marker=u'o')

    # Additional plot tweaks
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Distribution - KMeans Algorithm')

""" Notes on the comparison between K-Means x DBSCAN
    . K-Means inconsistency in the formed clusters (due to the initial centorid guess)
    . Superior capability of the DBSCAN algorithm in detecting outliers and prevent cluster overlapping
"""
