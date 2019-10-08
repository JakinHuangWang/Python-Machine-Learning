import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification, confusion_matrix
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pylab as plt
import itertools

churn_df = pd.read_csv("ChurnData.csv")
churn_df['churn'] = churn_df['churn'].astype('int')
col = churn_df.columns
X = churn_df[col[:len(col)-1]].values
Y = churn_df[col[len(col)-1]].values

scaled_X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.25, random_state = 25)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

# Construct our linear regression model
LR = LogisticRegression(C=0.1, solver="newton-cg").fit(X_train, Y_train)
print(LR)

#predict our model
y_hat = LR.predict(X_test)
print(y_hat)
#first column is the probability of class 1, second column is the probability of class 0
y_hat_prob = LR.predict_proba(X_test)
print(y_hat_prob)
print("Jaccard Similarity Score:", jaccard_similarity_score(Y_test, y_hat))

def plot_confusion_matrix(test, predict, normalize=False, title='Confusion Matrix', colormap = plt.cm.Blues):
    if normalize:
        test = test.astype('float') / test.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
   
    print(test)
     
plot_confusion_matrix(Y_test, y_hat)
    