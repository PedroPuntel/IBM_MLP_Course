# 01/01/2020
# Pedro H. Puntel
# pedro.puntel@gmail.com
# IBM Introduction To Machine Learning with Python
# Recommender Systems - Content Based Example

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
movies_df.head()

#%% Splitting the movies genres in the "genre" column into a list (then into a vector) for further use
"""
    Since keeping genres in a list format isn't optimal for the content-based recommendation system technique,
    we will use the One Hot Encoding technique to convert the list of genres to a vector where each column
    corresponds to one possible value of the feature. This encoding is needed for feeding categorical data.
    
    In this case, we store every different genre in columns that contain either 1 or 0. 1 shows that a movie has
    that genre and 0 shows that it doesn't. Let's also store this dataframe in another variable since genres
    won't be important for our first recommendation system.
"""
movies_df["genres"] = movies_df.genres.str.split("|")
movies_with_genres_df = movies_df.copy()

# for every row in the dataframe, iterate through the list of genres and flag into the corresponding column
for i, j in movies_df.iterrows():
    for genre in j["genres"]:
        movies_with_genres_df.at[i, genre] = 1
        
# Converting NA's to zeros, indicating that the given movie doesn't have that specific genre
movies_with_genres_df = movies_with_genres_df.fillna(0)
        
#%% Droping the timestamp column in the ratings data frame since we won't need it
ratings_df = ratings_df.drop("timestamp", 1)

#%% Building an item-item recommendation system based on a given user input
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

#%% Extracting movies id's from the movie movies_df dataset and combining them with the ones in the user input
inputId = movies_df[movies_df["title"].isin(inputMovies["title"].tolist())]
inputMovies = pd.merge(inputId,inputMovies)
inputMovies = inputMovies.drop("genres", 1).drop("year", 1)

#%% Retrieving the properties associated with the movies watched by the user
userMovies = movies_with_genres_df[movies_with_genres_df["movieId"].isin(inputMovies["movieId"].tolist())]
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop("movieId",1).drop("title",1).drop("genres",1).drop("year",1)

#%% Generating the user's vectorized profile
userProfile = userGenreTable.transpose().dot(inputMovies["rating"])

#%% Extracting the genre table from the original dataframe
genreTable = movies_with_genres_df.set_index(movies_with_genres_df["movieId"])
genreTable = genreTable.drop("movieId",1).drop("title",1).drop("year",1).drop("genres",1)

#%% Weighted average of every movie based on the input profile and recommend the top 10 movies that most satisfy it
recommendation_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendation_df = recommendation_df.sort_values(ascending=False)

#%% Final recommendation table
movies_df.loc[movies_df["movieId"].isin(recommendation_df.head(10).keys())][["title","genres"]]

""" Advantages and Disadvantages of Content-Based Filtering

    Advantages
        Learns user's preferences
        Highly personalized for the user
        
    Disadvantages
        Doesn't take into account what others think of the item, so low quality item recommendations might happen
        Extracting data is not always intuitive
        Determining what characteristics of the item the user dislikes or likes is not always obvious
"""