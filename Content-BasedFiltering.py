import numpy as np
import pandas as pd

movies_df = pd.read_csv('ml-latest/movies.csv')
ratings_df = pd.read_csv('ml-latest/ratings.csv')

print(movies_df.head())

movies_df['year'] = movies_df['year'].str.extract('\d\d\d\d', expand=False)