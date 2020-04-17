# 15/12/2019
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Hierarchical Clustering Example

""" This example will be looking at Agglomerative Hierarchical Clustering (bottom up approach),
    using Complete Linkage as the Linkage Criteria.
"""

#%% Modules
import numpy as np 

#%% Generating a random set of data
from sklearn.datasets.samples_generator import make_blobs
np.random.seed(0)
x1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

#%% Plotting the data
from matplotlib import pyplot as plt 
plt.scatter(x1[:, 0], y1[:], marker='o')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Distribution of the randomly generated data')

#%% Fitting the Hierarchical clustering using the complete linkage approach
# >> Complete linkage minimizes the maximum distance between all observations
from sklearn.cluster import AgglomerativeClustering
cplt_linkage_fit = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="complete")
cplt_linkage_fit.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method"complete", metric="euclidean")
cplt_linkage_fit_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clustered Data Distribution - Complete Linkage Approach')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(cplt_linkage_fit.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

#%% Fitting an Hierarchical clustering model usign average linkage approach
# >> Average linkage minimizes the average distance of each observation
avg_linkage_fit = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="average")
avg_linkage_fit.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method="average", metric="euclidean")
avg_linkage_fit_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clustered Data Distribution - Average Linkage Approach')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(avg_linkage_fit.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

#%% Fitting an Hierarchical clustering model usign Ward linkage approach
# >> Ward linkage minimizes the variance of the clusters beign merged
ward_linkage_fit = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
ward_linkage_fit.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method="ward", metric="euclidean")
ward_linkage_fit_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clustered Data Distribution - Ward Linkage Approach')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(ward_linkage_fit.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

#%%
""" It is curious to see that, without changing the affinity measure, the cluster
    assignment was pretty much the same between the different linkage criteria.
    
    We shall now do a different comparison instead : let's examine the effect of
    the affinity measure an the cluster assignment.
"""

# Euclidean dissimilarity/affinity measure
from sklearn.cluster import AgglomerativeClustering
cmplt_linkage_euclidean = AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete")
cmplt_linkage_euclidean.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method="complete", metric="euclidean")
cmplt_linkage_euclidean_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clusteres Distribution - (Complete Linkage | Euclidean Measure)')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(cmplt_linkage_euclidean.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

# Cosine disssimilarity/affinity measure
from sklearn.cluster import AgglomerativeClustering
cmplt_linkage_cosine = AgglomerativeClustering(n_clusters=4,affinity="cosine",linkage="complete")
cmplt_linkage_cosine.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method="complete", metric="cosine")
cmplt_linkage_cosine_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clusters Distribution - (Complete Linkage | Cosine Measure)')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(cmplt_linkage_cosine.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

# Manhattan disssimilarity/affinity measure
from sklearn.cluster import AgglomerativeClustering
cmplt_linkage_manhattan = AgglomerativeClustering(n_clusters=4,affinity="manhattan",linkage="complete")
cmplt_linkage_manhattan.fit(x1, y1)

# Dendogram
from scipy.spatial import distance_matrix as dist
from scipy.cluster import hierarchy as h
dist_matrix = dist(x1, x1)
z = h.linkage(y=dist_matrix, method="complete", metric="cityblock") # cityblock == manhattan
cmplt_linkage_manhattan_dendogram = h.dendrogram(z)

# Plotting clusters
plt.figure(figsize=(6,4))
plt.title('Clusters Distribution - (Complete Linkage | Manhattan Measure)')
plt.grid(axis='y', alpha=0.75)
plt.grid(axis='x', alpha=0.75)
plt.xlabel('X')
plt.ylabel('Y')
x_min, x_max = np.min(x1, axis=0), np.max(x1, axis=0)
x1 = (x1 - x_min) / (x_max - x_min)

for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(cmplt_linkage_manhattan.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

plt.scatter(x1[:, 0], x1[:, 1], marker='.')
plt.show()

""" Conclusion:
    
    Assigning different afinity measures to the hierarchical clustering algorithm, at
    first glance, doesn't seems to make much difference regardless the cluster assignment
    of the data.
    
    The only observed difference seems to be on the "height" or "cutting point" in which
    the clusters are formed. At the time of this writing, i would say that picking an
    affinity measure that "quickly clusters" the data (i.e affinity measures with low
    y-axis range) would be better, since the algorithm could "quickly" identify similar
    objects.
"""

#%%
""" Clustering on Vehicle dataset

    An automobile manufacturer has developed prototypes for a new vehicle. Before introducing the new model
    into its range, the manufacturer wants to determine which existing vehicles on the market are most like
    the prototypes in a sense that which models they will be competing against.

    Our objective here, is to use clustering methods, to find the most distinctive clusters of vehicles.
    It will summarize the existing vehicles and help manufacture to make decision about new models simply.
"""

# Fetching the example data from IBM server
import wget
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv"
file = wget.download(url)    

# Reding the data
import pandas as pd
cars_df = pd.read_csv("cars_clus.csv")
cars_df.shape
print(cars_df.head(5))

# Data cleaning
cars_df[['sales', 'resale', 'type', 'price', 'engine_s','horsepow',
         'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
         'mpg', 'lnsales']] = cars_df[['sales', 'resale', 'type',
                                       'price', 'engine_s',
                                       'horsepow', 'wheelbas',
                                       'width', 'length',
                                       'curb_wgt', 'fuel_cap',
                                       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
cars_df = cars_df.dropna()
cars_df = cars_df.reset_index(drop=True)

# Feature selection and scaling the dataset
from sklearn.preprocessing import MinMaxScaler
car_featureset = cars_df[['engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']]
x = car_featureset.values
min_max_scaler = MinMaxScaler()
car_featureset = min_max_scaler.fit_transform(x)
car_featureset[0:5]

#%% Agglomerative Hierarchical Clsutering with Complete linkage and Euclidean Affinity measure (using Scipy)
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy 
D = distance_matrix(car_featureset, car_featureset)
scipy_cars_linkage = hierarchy.linkage(y=D,method="complete",metric="euclidean")
scipy_cars_distance_clustering = fcluster(Z=scipy_cars_linkage,t=3,criterion="distance") # cluster cutting line (thershold)
scipy_cars_max_fixed_clustering = fcluster(scipy_cars_linkage,t=5,criterion="maxclust") # number of clusters

# Visualizing
import pylab
fig = pylab.figure(figsize=(18,50))

def llf(id):
    return '[%s %s %s]' % (cars_df['manufact'][id], cars_df['model'][id], int(float(cars_df['type'][id])) )   

cars_dendogram = hierarchy.dendrogram(scipy_cars_linkage,
                                      leaf_label_func=llf,
                                      leaf_rotation=0,
                                      leaf_font_size=12,
                                      orientation='right')

#%% Agglomerative Hierarchical Clsutering with Complete linkage and Euclidean Affinity measure (using Scikit-Learn)
from sklearn.cluster import AgglomerativeClustering
scikit_cars_linkage = AgglomerativeClustering(n_clusters=5, linkage='complete')
scikit_cars_linkage.fit(car_featureset)
scikit_cars_linkage.labels_

# Adding a new field to the dataframe to indicate de cars's beloging to a cluster
cars_df["cluster"] = scikit_cars_linkage.labels_
cars_df.head(5)

# Plotting each cluster's distribution of "horsepower x miles per gallon"
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt 
n_clusters = max(scikit_cars_linkage.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))
plt.figure(figsize=(16,14))

# For each color associated with each cluster
for color, label in zip(colors, cluster_labels):
    
    # Gather only cars which belong to the i-th cluster
    subset = cars_df[cars_df.cluster == label]
    
    # Assign their plot labels
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation=25) 
     
    # Plot the scatterplot  
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)

# Additional plot tweaks    
plt.legend()
plt.title("Analysing cluster distribution regardless horsepower and mpg")
plt.xlabel('horsepow')
plt.ylabel('mpg')

#%%
""" Inspecting the cluster distribution regardless the cars category (truck or car)

    As we can see, we are seeing the distribution of each cluster using the scatter plot, but it is not very clear
    where is the centroid of each cluster. Moreover, there are 2 types of vehicles in our dataset, "truck" and "car"
    (specified in the 'type' column). So, we use them to distinguish the classes, and then summarize the cluster.
"""

# Counting the number of cars and trucks in each cluster
cars_df.groupby(['cluster','type'])['cluster'].count()

# Now we look at each cluster
aggr_cars_df = cars_df.groupby(['cluster','type'])['horsepow','engine_s','mpg','price'].mean()

""" It is obvious that we have 3 main clusters with the majority of vehicles in those.

    Cars:
        Cluster 1: with almost high mpg, and low in horsepower.
        Cluster 2: with good mpg and horsepower, but higher price than average.
        Cluster 3: with low mpg, high horsepower, highest price.

    Trucks:
        Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
        Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
        Cluster 3: with good mpg and horsepower, low price.

    We did not use type and price of cars in the clustering process, but we could forge the clusters and discriminate
    usign this information with quite high accuracy.
"""

# Plotting
plt.figure(figsize=(16,10))

for color, label in zip(colors, cluster_labels):
    subset = aggr_cars_df.loc[(label,),]
    
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
        
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
    
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
