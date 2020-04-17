# 23/11/2019
# Pedro H. Sodré Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Decision Trees - Example

# Modules
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Fetches the example data from IBM server (already done)
import wget
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"
data = wget.download(url)


# Loads the data
data_path = "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts\\drug200.csv"
data = pd.read_csv(data_path, delimiter = ",")

# Defines separetely the explanatory/response variables
explanatory_data = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
response_data = data["Drug"].values

# Coerces the explanatory categorical variables to dummy variables
from sklearn import preprocessing

dummy_sex = preprocessing.LabelEncoder()
dummy_sex.fit(['F','M'])
explanatory_data[:,1] = dummy_sex.transform(explanatory_data[:,1]) 

dummy_bp = preprocessing.LabelEncoder()
dummy_bp.fit(['LOW', 'NORMAL', 'HIGH'])
explanatory_data[:,2] = dummy_bp.transform(explanatory_data[:,2])

dummy_chol = preprocessing.LabelEncoder()
dummy_chol.fit(['NORMAL', 'HIGH'])
explanatory_data[:,3] = dummy_chol.transform(explanatory_data[:,3])

# Allocates the training and test data
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(explanatory_data,
                                                                response_data,
                                                                test_size = 0.3,
                                                                random_state = 3)

# Fits the Decision-Tree model
drug_tree_model = DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
drug_tree_model.fit(X_trainset, y_trainset)

# Predicts on the testing dataset
yhat_tree_model = drug_tree_model.predict(X_testset)

# sklearn's model accuracy metric (ratio of correct predictions over the total)
from sklearn import metrics
metrics.accuracy_score(y_testset, yhat_tree_model)

# Model accuracy calculation by hand
dummy_ypred = preprocessing.LabelEncoder()
dummy_ypred.fit(["drugA","drugB","drugC","drugX","drugY"])
yhat_tree_model = dummy_ypred.transform(yhat_tree_model)

dummy_ytrue = preprocessing.LabelEncoder()
dummy_ytrue.fit(["drugA","drugB","drugC","drugX","drugY"])
y_testset = dummy_ytrue.transform(y_testset)

correct_predictions = 0
for i in np.arange(len(yhat_tree_model)):
    if yhat_tree_model[i] == y_testset[i]:
        correct_predictions += 1

correct_predictions/len(y_testset)
    
# Visualizing the Tree
import pydotplus
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz

dot_data = StringIO()
export_graphviz(drug_tree_model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) ## Doesn't work :-/
