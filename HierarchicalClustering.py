import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn import preprocessing
import matplotlib.cm as cm

X1, Y1 = make_blobs(n_samples=150, centers=[[1,  1], [2, 2], [6, 6], [-3, -4], [7, 9]], cluster_std=0.9)
print(X1.shape, Y1.shape)
print(len(X1[:,0]))
plt.scatter(X1[:, 0], X1[:, 1], c=cm.rainbow(np.linspace(0, 1, len(X1[:, 0 ]))))
plt.show()
agglom = AgglomerativeClustering(n_clusters=5, linkage='average')
agglom.fit(X1, Y1)

for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(agglom.labels_[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

dist_matrix = distance_matrix(X1, X1)
dist_matrix = preprocessing.normalize(dist_matrix)
print(dist_matrix.shape)
Z = hierarchy.linkage(dist_matrix, method='complete')
dendro = hierarchy.dendrogram(Z)
plt.show()

#The Automobile Part
df = pd.read_csv('cars_clus.csv')
print(df)
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print(df)
print ("Shape of dataset after cleaning: ", df.size)

featureset = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
features_MMS = MMS.fit_transform(featureset.values)
print(features_MMS)

import scipy
Dist = scipy.zeros((features_MMS.shape[0], features_MMS.shape[0]))
for i in range(Dist.shape[0]):
    for j in range(Dist.shape[0]):
        Dist[i, j] = scipy.spatial.distance.euclidean(features_MMS[i], features_MMS[j])
        
print(Dist)

Z = hierarchy.linkage(Dist, method='complete')
def llf(id):
    return '[%s %s %s]' % (df['manufact'][id], df['model'][id], int(float(df['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_font_size =4, orientation = 'right')
plt.show()