# 23/11/2019
# Pedro H. Sodré Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Non Linear Regression - Example

# Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Fetching data from IBM server
import wget
%cd "C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts"
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv"
data = wget.download(url)

# Importing the data
china_gdp = pd.read_csv(data)
china_gdp.head(10)
china_gdp.describe()

# Visually inspecting the data
plt.figure(figsize=(8,5))
plt.scatter(china_gdp["Year"].values,china_gdp["Value"].values,color="green",alpha=0.7)
plt.grid(axis="y",alpha=0.75)
plt.ylabel("China's GDP")
plt.xlabel('Years')
plt.title("Scatterplot: China's GDP Growth (1960 - 2010)")
plt.show()

# From the visual inspection, it seems that a exponential fucntion is a good approximation
# . We could linearize our model by taking the logarithm :-)
x = np.arange(-5.0, 5.0, 0.1)
y = np.exp(x)
plt.plot(x,y,color="red",alpha=0.7) 
plt.grid(axis="y",alpha=0.75)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.title("Exponential Function")
plt.show()

# Builiding the model
import statistics as stat
from scipy.optimize import curve_fit
def exponential(x, beta1, beta2):
    y = beta1 + beta2*np.exp(x)
    return y
years_std = (china_gdp["Year"].values-stat.mean(china_gdp["Year"].values))/(stat.stdev(china_gdp["Value"].values))
gdp_std = (china_gdp["Value"].values-stat.mean(china_gdp["Value"].values))/(stat.stdev(china_gdp["Value"].values))

# Fitting the model
exp_fit = curve_fit(exponential, years_std, gdp_std)
print("intercept = %f", exp_fit[0])
print("Beta 1 = %f", exp_fit[1])

# Visualizing the fitted model
x = np.linspace(1960, 2015, 55)
x = (x-stat.mean(x))/stat.stdev(x)
y = exponential(x, exp_fit[0][1], exp_fit[1][0][0])
plt.figure(figsize=(8,5))
plt.plot(x,y, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title("Inspection: Exponential model fit")
plt.show()
