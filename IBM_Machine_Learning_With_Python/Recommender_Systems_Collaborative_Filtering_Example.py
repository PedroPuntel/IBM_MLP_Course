# 01/01/2020
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Recommender Systems - Collaborative Filtering Example

#%% Modules
import wget
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
from math import sqrt

#%% Fetching the data
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip"
file = wget.download(url)
with zipfile.ZipFile(file, "r") as zip_ref:
    zip_ref.extractall("C:\\Users\\pedro\\Downloads\\IBM_Machine_Learning_With_Python\\Scripts")

#%% Reading the data
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

#%% Removing the year from the movies title and storing it in a new column called "year"
movies_df["year"] = movies_df.title.str.extract("(\(\d\d\d\d\))", expand=False)
movies_df["year"] = movies_df.year.str.extract("(\d\d\d\d)", expand=False)
movies_df["title"] = movies_df.title.str.replace("(\(\d\d\d\d\))", "")
movies_df["title"] = movies_df["title"].apply(lambda x: x.strip())
movies_df = movies_df.drop("genres",1)
movies_df.head()

#%% Droping the timestamp column in the ratings data frame since we won't need it
ratings_df = ratings_df.drop("timestamp", 1)

#%% Building an User-User Collaborative Filtering based on a user input
""" Collaborative Filtering uses other users to recommend items to the input user. It attempts to find users that
    have similar preferences and opinions as the input and then recommends items that they have liked to the input.
    
    There are several methods of finding similar users (Even some making use of Machine Learning), and the one we 
    will be using here is going to be based on the Pearson Correlation Function.

    The process for creating a User Based recommendation system is as follows:

        .Select a user with the movies the user has watched
        .Based on his rating to movies, find the top X neighbours
        .Get the watched movie record of the user for each neighbour.
        .Calculate a similarity score using some formula
        .Recommend the items with the highest score
"""
# User movie ratings
user_input = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5},
            {"title":"Matrix, The","rating":7.8},
            {"title":"Ace Ventura: When Nature Calls","rating":6},
            {"title":"Nixon","rating":9},
            {"title":"Casino","rating":8.7},
            {"title":"Sense and Sensibility","rating":4.1},
            {"title":"American President, The","rating":7.1},
            {"title":"Pocahontas","rating":3.2},
            {"title":"Mortal Kombat","rating":8.2},
            {"title":"Babe","rating":1.3}
    ]
inputMovies = pd.DataFrame(user_input)

# Adding movieId to user input
inputId = movies_df[movies_df["title"].isin(inputMovies["title"].tolist())]
inputMovies = pd.merge(inputId,inputMovies)
inputMovies = inputMovies.drop("year",1)

# Obtaining the top 100 users who have watched the same movies as our target user
userSubset = ratings_df[ratings_df["movieId"].isin(inputMovies["movieId"].tolist())]

# Grouping the rows of the userSubset data frame by userId
userSetSubgroup = userSubset.groupby(["userId"])

# Sorting the grouped userSubset data frame in such way that similar users to our target user are given higher
# priority, for example : "XXX has 15 movies in common with YYY and ZZZ has only 5. XXX will have priority".
userSetSubgroup = sorted(userSetSubgroup, key=lambda x: len(x[1]), reverse=True)

# Working only with the Top 100 groups of similar users (doesn't make sense to consier all of them)
userSetSubgroup = userSetSubgroup[0:100]

# Calculating the similarity of users to the input user usign the Pearson Correlation Coefficient
#
# . (?) Why Pearson Correlation Coefficient ?
#
# "Pearson correlation is invariant to linear transformations, (i.e r(X,Y) == r(X, 2*Y+3)).
#  This is a pretty important property in recommendation systems because, for example, two users might rate
#  two series of items totally different in terms of absolute rates, but they would be similar users
#  (i.e. with similar ideas) with similar rates in various scales".

# We shall store the similarity scores in a dictionary whose key is the userId and value is the r(x,y)
from scipy.stats import pearsonr as r
pearsonCorrelationDict = {}

# For every user group in our subset
for userid, group in userSetSubgroup:
    
    # Sorting the input user and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
       
    # Movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    
    # User input movie ratings in list format to facilitate calculations
    tempRatingList = temp_df['rating'].tolist()
    
    # User group movie ratings also in list format
    tempGroupList = group['rating'].tolist()
    
    # Correlation score
    pearsonCorrelationDict[userid] = r(tempRatingList, tempGroupList)[0]

#%% Exploring the created dictionary
corr_df = pd.DataFrame.from_dict(pearsonCorrelationDict, orient="index")
corr_df.columns = ["similarity_index"]
corr_df["userId"] = corr_df.index
corr_df.index = range(len(corr_df))
corr_df.head()

#%% Top 50 similar users to input user
topUsers_df = corr_df.sort_values(by="similarity_index", ascending=False)[0:50]
topUsers_df.head()

#%% Now we're going to extract the movies ratings of the selected most similar users
topUsers_ratings = topUsers_df.merge(ratings_df, left_on="userId", right_on="userId", how="inner")
topUsers_ratings.head()

#%% Computing the weighted average of the ratings of the movies, using the Pearson Correlation as the weight
topUsers_ratings["weighted_rating"] = topUsers_ratings["similarity_index"]*topUsers_ratings["rating"]
topUsers_ratings = topUsers_ratings.groupby("movieId").sum()[["similarity_index", "weighted_rating"]]
topUsers_ratings.columns = ["sum_similarity_index","sum_weighted_rating"]
topUsers_ratings.head()

#%% Building the recommendation dataset
recommendation_df = pd.DataFrame()
recommendation_df["weighted_score"] = topUsers_ratings["sum_weighted_rating"]/topUsers_ratings["sum_similarity_index"]
recommendation_df["movieId"] = topUsers_ratings.index
recommendation_df.head()

#%% Retrieving the Top 20 movies the algorithm recommended
recommendation_df = recommendation_df.sort_values(by="weighted_score", ascending=False)
movies_df.loc[movies_df["movieId"].isin(recommendation_df.head(20)["movieId"].tolist())]

""" Advantages and Disadvantages of Collaborative Filtering

    Advantages
        Takes other user's ratings into consideration
        Doesn't need to study or extract information from the recommended item
        Adapts to the user's interests which might change over time
    
    Disadvantages
        Approximation function can be slow
        There might be a low of amount of users to approximate
        Privacy issues when trying to learn the user's preferences
"""
