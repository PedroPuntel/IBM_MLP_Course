# 24/11/2019
# Pedro H. Sodré Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# KNN - Example

# Modules
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter

# Fetching the data from IBM server
import wget
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv"
data = wget.download(url)

# Importing the data
telecom_df = pd.read_csv("teleCust1000t.csv")
telecom_df.head(5)
telecom_df.describe()
telecom_df.columns

# Explorig the variables
plt.bar(np.arange(len(np.unique(telecom_df["region"])))+1, telecom_df["region"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Region')
plt.ylabel('Frequency')
plt.title('Distribution of clients by region')
plt.xticks(np.arange(len(np.unique(telecom_df["region"])))+1, np.unique(telecom_df["region"]))

plt.hist(x=telecom_df["tenure"], bins="auto", color='red', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.title("Distribution of client's tenure")

plt.hist(x=telecom_df["age"], bins="auto", color='puprle', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title("Distribution of client's age")

plt.bar(np.arange(len(np.unique(telecom_df["marital"]))), telecom_df["marital"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Married')
plt.ylabel('Frequency')
plt.title('Distribution of married clients')
plt.xticks(np.arange(len(np.unique(telecom_df["marital"]))), ["no","yes"])

plt.hist(x=telecom_df["income"], bins="auto", color='green', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title("Distribution of client's income")

plt.bar(np.arange(len(np.unique(telecom_df["ed"])))+1, telecom_df["ed"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Years of Education')
plt.ylabel('Frequency')
plt.title("Distribution of client's years of education")
plt.xticks(np.arange(len(np.unique(telecom_df["ed"])))+1, np.unique(telecom_df["ed"]))

plt.hist(x=telecom_df["employ"], bins="auto", color='yellow', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Years employed')
plt.ylabel('Frequency')
plt.title("Distribution of client's years employed")

plt.bar(np.arange(len(np.unique(telecom_df["retire"]))), telecom_df["retire"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Retirement')
plt.ylabel('Frequency')
plt.title("Distribution of retired clients")
plt.xticks(np.arange(len(np.unique(telecom_df["retire"]))), ["Not retired","Retired"])

plt.bar(np.arange(len(np.unique(telecom_df["gender"]))), telecom_df["gender"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title("Distribution of client's gender")
plt.xticks(np.arange(len(np.unique(telecom_df["gender"]))),["Men","Women"])

plt.bar(np.arange(len(np.unique(telecom_df["reside"])))+1, telecom_df["reside"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Residement')
plt.ylabel('Frequency')
plt.title("Distribution of client's residement")
plt.xticks(np.arange(len(np.unique(telecom_df["reside"])))+1, np.unique(telecom_df["reside"]))

plt.bar(np.arange(len(np.unique(telecom_df["custcat"])))+1, telecom_df["custcat"].value_counts())
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Custumer Category')
plt.ylabel('Frequency')
plt.title("Distribution of custumer category")
plt.xticks(np.arange(len(np.unique(telecom_df["custcat"])))+1, ["Basic","E-Service","Plus","Total"])

# Splitting our data in explanatory/response variables
explanatory_df = telecom_df[['region', 'tenure','age', 'marital', 'address',
                            'income', 'ed','employ','retire', 'gender', 'reside']].values
response_df = telecom_df["custcat"].values

# Standardizing the data (good practice since KNN takes the euclidean distance)
from sklearn import preprocessing
explanatory_df = preprocessing.StandardScaler().fit(explanatory_df).transform(explanatory_df)

# Allocating our train/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(explanatory_df, response_df, test_size=0.2, random_state=4)

# Implementing the KNN algorithm (assuming k = 4)
from sklearn.neighbors import KNeighborsClassifier
k = 4
knn_4_fit = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

# Predicting with the previuously fitted model
custcat_hat = knn_4_fit.predict(X_test)

# Accuracy evaluation (equivalent to the jaccard index)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, knn_4_fit.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, custcat_hat))

# Computing the model with the optimal number of k
tot_k = 10
mean_acc = np.zeros((tot_k-1))
std_acc = np.zeros((tot_k-1))

for i in range(1, tot_k):
    # Computes the knn model with different values for "k" and calculates its accuracy scores
    knn_fit = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    custcat_fit = knn_fit.predict(X_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, custcat_fit)
    std_acc[i-1] = np.std(custcat_fit == y_test)/np.sqrt(custcat_fit.shape[0])
        
print("The best accuracy was", mean_acc.max(), "with k =", mean_acc.argmax()+1) 

# Plotting the accuracy of each model
plt.plot(range(1,tot_k), mean_acc,'g')
plt.fill_between(range(1,tot_k), mean_acc - 1*std_acc, mean_acc + 1*std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3*std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.title("Optimal values for K")
plt.show()
