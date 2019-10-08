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

kmeans = KMeans(n_clusters=K)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

print(centroids)
print(labels)

colors = map(lambda x: colormap[x+1], labels)
print(colors)
plt.scatter(df['X'], df['Y'], c=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colormap[idx+1])
plt.show()