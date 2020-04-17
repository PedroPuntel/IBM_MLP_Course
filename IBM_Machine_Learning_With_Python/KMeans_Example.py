# 04/12/2019
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# K-Means Example

#%% Required modules
import random 
import numpy as np 
import matplotlib.pyplot as plt 

#%% Generating a random dataset
from sklearn.datasets.samples_generator import make_blobs 
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

#%% Scatter plot of the randomly generated data
plt.scatter(X[:,0], X[:,1], marker=".")

"""
The K-Means class has many parameters that can be used, but we will be using these three:

    . init: Initialization method of the centroids. Value will be: "k-means++"
      k-means++: Selects initial cluster centers for k-mean clustering to speed convergence.
      
    . n_clusters: The number of clusters to form as well as the number of centroids to generate.
      Value will be: 4 (since we have 4 centers)
      
    . n_init: Number of time the k-means algorithm will be run with different centroid seeds.
      The final results will be the best output of n_init consecutive runs in terms of inertia.     
"""

#%% Setting up and fitting the model
from sklearn.cluster import KMeans 
k_means = KMeans(init = "k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

#%% Retrieving the labels from each point in the model
k_means_labels = k_means.labels_

#%% Retrieving the coordinates from each cluster center in the model
k_means_centers = k_means.cluster_centers_

#%% Creating the visual plot
fig = plt.figure(figsize=(6, 4)) # Initialize the plot with the specified dimensions.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels)))) # Maps colors for each unique l
ax = fig.add_subplot(1, 1, 1) # Creates the plot object
ax.set_title('KMeans') # Title of the plot
ax.set_xticks(())      # Remove x-axis ticks
ax.set_yticks(())      # Remove y-axis ticks

""" Plots the data points and centroids. "k" will range from 0-3, which will
    match the possible clusters that each data point is in.
"""
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    """list of all data points, where the data poitns that are 
    in the cluster are labeled as true, else they are labeled false."""
    my_members = (k_means_labels == k)
    
    """Define the centroid, or cluster center"""
    cluster_center = k_means_centers[k]
    
    """Plots the datapoints with color col"""
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    """Plots the centroids with specified color, but with a darker outline"""
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

plt.show()

"""
Customer Segmentation with K-Means

Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data.
Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar
characteristics. It is a significant strategy as a business can target these specific groups of customers and
effectively allocate marketing resources. For example, one group might contain customers who are high-profit and
low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining
those customers. Another group might include customers from non-profit organizations. And so on.
"""

#%% Fetching the example data from IBM server
import wget
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv"
file = wget.download(url)    

#%% Loading the example data
import pandas as pd
customers_df = pd.read_csv("Cust_Segmentation.csv")

#%% Removing categorial variables from the dataset since euclidean distance doesn't apply for categorical data
customers_df = customers_df.drop("Address", axis=1)
customers_df.head(5)

#%% Normalizing the customers data set over the standard deviation
from sklearn.preprocessing import StandardScaler
X = customers_df.values[:, 1:]
X = np.nan_to_num(X)
X = StandardScaler().fit_transform(X)

#%% Modeling (usign k = 3)
k_means_costumer = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means_costumer.fit(X)

#%% Once again, retrieving points labels from the model and assign then to wach row in the dataframe
customer_labels = k_means_costumer.labels_
customers_df["Cluster_ID"] = customer_labels
customers_df.head(5)

#%% Extracting insights from data
customers_centers = customers_df.groupby("Cluster_ID").mean() ## Centroid values of each cluster

#%%
""" Inspecting the distribution of customers based on their age and income """
plot_area = np.pi * (X[:,1])**2
plt.scatter(X[:,0], X[:,3], s=plot_area, c=customer_labels.astype(np.float), alpha=0.5)
plt.xlabel("Age", fontsize=18)
plt.ylabel("Income", fontsize=18)
plt.show()

#%%
""" Inspecting the distribution of costumers based on age, income and education """
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=customer_labels.astype(np.float))

#%% Cluster Evaluation Metrics

