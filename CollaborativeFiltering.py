import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

movies_df = pd.read_csv('Collaborative Filtering/movies.csv')
ratings_df = pd.read_csv('Collaborative Filtering/ratings.csv')

movies_df['year'] = movies_df['title'].str.extract('(\(\d\d\d\d\))', expand=False).str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df['title'].str.replace('(\(\d\d\d\d\))', '')
movies_df = movies_df.drop('genres', 1)
ratings_df = ratings_df.drop('timestamp', 1)

userProfile = pd.DataFrame([
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
])


print(movies_df['title'].isin(userProfile['title']))
