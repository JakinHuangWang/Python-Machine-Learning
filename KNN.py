import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv("teleCust1000t.csv")
c = list(df.columns)
x = df[c[:len(c)-1]].values
y = df[c[len(c)-1]].values
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

#Now we do train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)


#Now we do the KNN-Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
k = 2
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)

#Then we predict the value
y_hat = neigh.predict(x_test)

#Finally we evaluates its accuracy
from sklearn import metrics

accu_Lst = []
for i in range(1, 100):
    neighi = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
    y_hat = neighi.predict(x_test)
    accu_Lst.append(metrics.accuracy_score(y_test, y_hat))
    print("F1 Score:", metrics.f1_score(y_test, y_hat, average=None))
print("Max Accuracy:", max(accu_Lst))   
plt.plot(np.arange(1, 100), accu_Lst, '-g')
plt.show()

from sklearn.metrics import confusion_matrix


    

