# 30/11/2019
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Support Vector Machine (SVM's) Example

"""
In this example, we'll build and train an SVM model using human cell records, aiming to classify cells
to whether they are benign or malignant. The dataset consists of several hundred human cell sample records,
each of which contains the values of a set of cell characteristics.

SVM works by mapping data to a high-dimensional feature space so that data points can be categorized,
even when the data are not otherwise linearly separable. A separator between the categories is found,
then the data are transformed in such a way that the separator could be drawn as a hyperplane.

Following this, characteristics of new data can be used to predict the group to which a new record should
belong.
"""

# Modules
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Fetching the data from IBM server
import wget
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv"
file = wget.download(url)

# Loading the data
cell_df = pd.read_csv("cell_samples.csv")
cell_df.head(10)
cell_df.shape

# Distribuition of cell's classes (malign/beningn) based on Clump thickness and Uniformity cell size
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

# Cleaning the data
cell_df.dtypes # BareNuc seems awkward (why it's not recognized as numeric ?)
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

# Splitting the data in our explanatory/response datasets
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
cell_df['Class'] = cell_df['Class'].astype('int')
X = np.asarray(feature_df)
y = np.asarray(cell_df['Class'])

# Train/Test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train.shape
X_test.shape

"""
The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a
higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the
kernel function, and can be of different types, such as:

    .Linear
    .Polynomial
    .Radial basis function (RBF)
    .Sigmoid
    
Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of
knowing which function performs best with any given dataset, we usually choose different functions in turn and compare
the results.
"""

# Fitting an SVM model using the RBF kernel function
from sklearn import svm
svm_rbf_fit = svm.SVC(kernel = "rbf")
svm_rbf_fit.fit(X_train, y_train)

# Predicting
svm_cancer_predict = svm_rbf_fit.predict(X_test)

# Computing several model evaluation metrics
from sklearn.metrics import jaccard_score, f1_score, classification_report
f1_score(y_test, svm_cancer_predict, average="weighted")
jaccard_score(y_true = y_test, y_pred = svm_cancer_predict, average="weighted")
classification_report(y_test, svm_cancer_predict, output_dict=True)

# Builds and plots the confusion matrix of the model
import itertools
from sklearn.metrics import confusion_matrix
def plot_cm(cm_obj, classes, cm_map=plt.cm.Blues, normalize=False, title="Confusion Matrix"):  
    """ Prints and plots a confusion matrix. Normalize the axis by tuning the 'normalize' parameter."""
    
    # Normalizes the matrix so that its values are between 0 and 1
    if normalize:
        cm_obj = cm_obj.astype("float") / cm_obj.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    # Prints the confusion matrix
    print(cm_obj)

    # Defines plot parameters
    plt.imshow(X = cm_obj, interpolation='nearest', cmap=cm_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Number output format
    fmt = '.2f' if normalize else 'd'
    
    # Threshold for color mapping
    thresh = cm_obj.max() / 2.

    # Crestes the confusion matrix
    for i, j in itertools.product(range(cm_obj.shape[0]), range(cm_obj.shape[1])):
        plt.text (
            j, i,
            format(cm_obj[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm_obj[i, j] > thresh else "black"
        )
        
    # plot layout
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
np.set_printoptions(precision=3)
cnf_matrix = confusion_matrix(y_test, svm_cancer_predict, labels=[2,4])
plt.figure()
plot_cm(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize=True,title='Confusion matrix')
