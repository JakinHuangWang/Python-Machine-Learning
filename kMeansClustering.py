import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy

df = pd.read_csv("drug200.csv")
Y = np.asarray(df['Na_to_K'])
X = np.asarray(df['Age'])
df = pd.DataFrame({'X':X, 'Y':Y})

K = 5
centroids = {i+1:[np.random.random_sample() * 80, np.random.random_sample()*40] for i in range(K)}
colormap = {1:'red', 2:'green', 3:'blue', 4:'yellow', 5:'cyan'}

def assignment(df, centroids):
    for i in centroids:
        df['distance_from_{}'.format(i)] = np.sqrt((df['X'] - centroids[i][0])**2 + (df['Y'] - centroids[i][1])**2)
    distance_cols = ['distance_from_{}'.format(i) for i in centroids]
    df['closest'] = df.loc[:, distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x : int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda i:colormap[i])
    return df

def update(centroids):
    for i in centroids:
        centroids[i][0] = np.mean(df[df['closest'] == i]['X'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['Y'])
    return centroids

df = assignment(df, centroids)
while True:
    closest_centroids = df['closest'].copy(deep=True)
    old_centroids = copy.deepcopy(centroids)
    centroids = update(centroids)  
    for i in centroids:
        plt.scatter(*centroids[i], c=colormap[i])    
    df = assignment(df, centroids)
    plt.scatter(df['X'], df['Y'], c=df['color'], alpha=0.5, edgecolors='k')
    plt.show()    
    if closest_centroids.equals(df['closest']):
        print("Optimized Result Generated!!")
        plt.scatter(df['X'], df['Y'], c=df['color'], alpha=0.5, edgecolors='k')
        for i in centroids:
            plt.scatter(*centroids[i], c=colormap[i])         
        plt.show()
        break
