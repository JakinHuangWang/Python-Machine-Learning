import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def createDataPoints(samples, centroidLocations, clusterDeviation):
    X, Y = make_blobs(n_samples=samples, centers=centroidLocations, cluster_std=clusterDeviation)
    X = StandardScaler().fit_transform(X)
    return X, Y
X, Y = createDataPoints(1500, [[4,3], [2,-1], [-1,4]], 0.5)

#Epsilon: The Radius, minimumSamples: The minimum points to be a core point, 
#-1 is noise other integers are the classes or called outliers
epsilon = 0.3
minimumSamples = 8
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(db.labels_)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
print(unique_labels)

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black' 
    class_member_mask = (db.labels_ == label)
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=color)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=color)
plt.show()

from sklearn.cluster import KMeans

KM = KMeans(n_clusters=3)
KM.fit(X)
labels = KM.labels_
centroids = KM.cluster_centers_
print(labels)
for label, color in zip(set(labels), colors):
    plt.scatter(X[labels == label, 0], X[labels==label, 1], c= color)
plt.show()
#for label, color in zip(labels, colors):
    
df = pd.read_csv('weather-stations20140101-20141231.csv')
print(df)
df = df[pd.notnull(df['Tm'])]
df = df.reset_index(drop = True)
print(df)

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations        

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()
#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
    my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()

