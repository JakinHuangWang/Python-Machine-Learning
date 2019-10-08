import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head(10))

ax = cell_df[cell_df['Class'] == 4][:].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
cell_df[cell_df['Class'] == 2][:].plot(kind='scatter', x ='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show()

print(cell_df.dtypes)
#Change to all int
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df = cell_df.astype('int')
print(cell_df.dtypes)

cols = cell_df.columns
X = np.asanyarray(cell_df[cols[1:len(cols)-1]])
Y = np.asanyarray(cell_df[cols[len(cols)-1]])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

plt.show()

#Now we transform our data to higher dimension using kernelling
from sklearn import svm
vec = svm.SVC(kernel='rbf')
vec.fit(X_train, Y_train)
print(vec)
y_hat = vec.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_hat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(Y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()

print(sklearn.metrics.f1_score(Y_test, y_hat, average = 'weighted'))
print(sklearn.metrics.jaccard_similarity_score(Y_test, y_hat))