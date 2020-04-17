# 23/11/2019
# Pedro H. Sodré Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Simple Linear Regression - Example

# Modules
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import wget
%matplotlib inline

# Fetching the data
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"
data = wget.download(url)

# Reading the data
df = pd.read_csv(data)
df.head()
df.describe()

# Explorig the data
import statistics
sub_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
sub_df.head(10)

# Exploring the "ENGINESIZE" variable
statistics.mean(sub_df["ENGINESIZE"]) ## aprox. 3.34
statistics.stdev(sub_df["ENGINESIZE"]) ## aprox 1.41
plt.hist(x=sub_df["ENGINESIZE"],bins="auto",color='blue',alpha=0.7,rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Egine Size')
plt.ylabel('Frequency')
plt.title('Histogram : ENGINESIZE variable')

# Exploring the linear relationship between ENGINESIZE x C02EMISSIONS
plt.scatter(x=sub_df["ENGINESIZE"],y=sub_df["CO2EMISSIONS"],color="blue",alpha=0.7)
plt.grid(axis="y",alpha=0.75)
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")
plt.title("Scatterplot: ENGINESIZE x CO2EMISSION")

# Exploring the "CYLINDERS" variable
statistics.mean(sub_df["CYLINDERS"]) ## aprox. 5.79
statistics.stdev(sub_df["CYLINDERS"]) ## aprox 1.79
plt.hist(x=sub_df["CYLINDERS"], bins="auto", color='red', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Cylinders')
plt.ylabel('Frequency')
plt.title('Histogram : CYLINDERS variable')

# Exploring the linear relationship between CYLINDERS x C02EMISSIONS
plt.scatter(x=sub_df["CYLINDERS"],y=sub_df["CO2EMISSIONS"],color="red",alpha=0.7)
plt.grid(axis="y",alpha=0.75)
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emission")
plt.title("Scatterplot: CYLINDERS x CO2EMISSION")

# Exploring the "FUELCONSUMPTION_COMB" variable
statistics.mean(sub_df["FUELCONSUMPTION_COMB"]) ## aprox. 11.58
statistics.stdev(sub_df["FUELCONSUMPTION_COMB"]) ## aprox. 3.48
plt.hist(x=sub_df["FUELCONSUMPTION_COMB"], bins="auto", color='green', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Fuel Consumption Combined')
plt.ylabel('Frequency')
plt.title('Histogram : FUELCONSUMPTION_COMB variable')

# Exploring the linear relationship between FUELCONSUMPTION_COMB x C02EMISSIONS
plt.scatter(x=sub_df["FUELCONSUMPTION_COMB"],y=sub_df["CO2EMISSIONS"],color="green",alpha=0.7)
plt.grid(axis="y",alpha=0.75)
plt.xlabel("Fuel Consumption Combined")
plt.ylabel("CO2 Emission")
plt.title("Scatterplot: FUELCONSUMPTION_COMB x CO2EMISSION")

# Exploring the "CO2EMISSIONS" variable
statistics.mean(sub_df["CO2EMISSIONS"]) ## aprox. 256.22
statistics.stdev(sub_df["CO2EMISSIONS"]) ## aprox. 63.37
plt.hist(x=sub_df["CO2EMISSIONS"], bins="auto", color='yellow', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('C02 Emssion')
plt.ylabel('Frequency')
plt.title('Histogram : CO2EMISSIONS variable')

# Allocating Train/Test data
rdm_rows_to_select = np.random.rand(len(df)) < 0.8 ## randomly selects the rows
train_df = df[rdm_rows_to_select] ## selects 80% of the data for training
test_df = df[~rdm_rows_to_select] ## selects 20% of the data for testing

# Inspectig the train data distribuition
# . As expected, it is similar to the "full" data distribuition (good sample)
plt.scatter(x=train_df.ENGINESIZE,y=train_df.CO2EMISSIONS,color="purple",alpha=0.7)
plt.grid(axis="y",alpha=0.75)
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
plt.title("Scatterplot: ENGINESIZE x CO2EMISSION")

# Fitting the model (Univariate)
from sklearn import linear_model
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train_df[['ENGINESIZE']])
train_y = np.asanyarray(train_df[['CO2EMISSIONS']])
regression.fit (train_x, train_y)
print ('Coefficients: ',regression.coef_)
print ('Intercept: ',regression.intercept_)

# Fitting the model (Multivariate)
# . Y ~ (D*A) + (D*B*X) + e
# . In which D is the matrix thar creates dummies for each cylinder size.

# Model evaluation
# . pip install statsmodels (for more robust statistical diagnostic tests)
# . https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0
from sklearn.metrics import r2_score
test_x = np.asanyarray(test_df[['ENGINESIZE']])
test_y = np.asanyarray(test_df[['CO2EMISSIONS']])
test_y_ = regression.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) ) # That ain't a good model...

