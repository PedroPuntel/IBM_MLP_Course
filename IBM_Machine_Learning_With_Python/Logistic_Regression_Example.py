# 30/11/2019
# Pedro H. Sodré Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning in Python
# Logistic Regression Example

# Modules
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Fetching example data from IBM server
import wget
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv"
file = wget.download(url)

# Loads the data
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()
churn_df.describe()
churn_df.columns

# Subsets the data according to the variables that will be used in the model 
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

# Encodes the response variable as binary
churn_df["churn"] = churn_df["churn"].astype("int")
churn_df.head()

# Defining our explanatory variables dataset and response vector
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']])
y = np.asarray(churn_df["churn"])

# Normalizing the dataset
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# Defining our train/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train.shape
X_test.shape

"""
Lets build our model using LogisticRegression from Scikit-learn package. This function implements logistic regression
and can use different numerical optimizers to find parameters. Extensive information about the pros and cons of these
optimizers if you search it in internet.

The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to
solve the overfitting problem in machine learning models. The "C" parameter indicates inverse of regularization
strength which must be a positive float. Smaller values specify stronger regularization.

>> https://towardsdatascience.com/regularization-an-important-concept-in-machine-learning-5891628907ea
"""

# Fitting the model
from sklearn.linear_model import LogisticRegression
logit_fit = LogisticRegression(C=0.01, solver="liblinear", fit_intercept=True).fit(X_train, y_train)
logit_fit 

# Predicting the churn state using the fitted model
churn_hat = logit_fit.predict(X_test)
churn_hat

# Calculating the probabilities of churn by a user
churn_prob = logit_fit.predict_proba(X_test)
churn_prob

# Computing several model evaluation metrics
from sklearn.metrics import jaccard_score, log_loss, classification_report
jaccard_score(y_test, churn_hat) # Not very good...
log_loss(y_test, churn_prob) # When the classifier is to predict the probability, performs better
print(classification_report(y_test, churn_hat)) # More accurate for the "0" class

# Creating an confusion matrix
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
    
# Visualizing
np.set_printoptions(precision=3)
cnf_matrix = confusion_matrix(y_test, churn_hat, labels=[1,0])
plt.figure()
plot_cm(cnf_matrix, classes=['churn=1','churn=0'], normalize=False, title='Confusion matrix')
plot_cm(cnf_matrix, classes=['churn=1','churn=0'], normalize=True, title='Confusion matrix')
