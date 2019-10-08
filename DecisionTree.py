import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
import matplotlib.pyplot as plt

#Decision Tree with teleCust1000t.csv
#df = pd.read_csv("teleCust1000t.csv")
#cols = df.columns
#X = df[cols[:len(cols)-1]].values
#Y = df[cols[len(cols)-1]].values
#X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 2)
#teleTree = DecisionTreeClassifier(criterion="entropy")
#teleTree.fit(X_train, Y_train)
#predTree = teleTree.predict(X_test)
#print(predTree)
#print("Accuracy Score:", sklearn.metrics.accuracy_score(Y_test, predTree))

#dot_data = StringIO()
#filename = "TeleTree.png"
#featureNames = df.columns[0:11]
#targetNames = df["custcat"].unique().tolist()
#out=tree.export_graphviz(teleTree, feature_names=featureNames, out_file=dot_data, filled=True,  special_characters=True,rotate=False)  
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100, 200))
#plt.imshow(img,interpolation='nearest')

##Decision Tree with drug200.csv

#df = pd.read_csv("drug200.csv")
#cols = df.columns
#X = df[cols[:len(cols)-1]].values
#Y = df[cols[len(cols)-1]].values

##Sklearn Decision Trees do not handle categorical variables. 
##But still we can convert these features to numerical values. 
##pandas.get_dummies() Convert categorical variable into dummy/indicator variables.

#encoder_sex = sklearn.preprocessing.LabelEncoder()
#encoder_sex.fit(['F', 'M'])
##So 0 represents F and 1 represents M

#X[:, 1] = encoder_sex.transform(X[:,1])

#encoder_BP = sklearn.preprocessing.LabelEncoder()
#encoder_BP.fit(['LOW', 'NORMAL', 'HIGH'])
##0->LOW, 1->NORMAL, 2->HIGH
#X[:, 2] = encoder_BP.transform(X[:,2])

#encoder_chol = sklearn.preprocessing.LabelEncoder()
#encoder_chol.fit(['NORMAL', 'HIGH'])
##0->NORMAL, 1->HIGH
#X[:, 3] = encoder_chol.transform(X[:, 3])

#X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.37, random_state = 3)

##Construct our Decision Tree
#drugTree = DecisionTreeClassifier(criterion="gini", max_depth=4)
#drugTree.fit(X_train, Y_train)

##After training our data, now we make predictions
#predTree = drugTree.predict(X_test)

##Calculate our score
#print("Accuracy Score:", sklearn.metrics.accuracy_score(Y_test, predTree))

#score = 0.0
#for i in range(len(Y_test)):
    #if Y_test[i] == predTree[i]:
        #score += 1.0
#print("Accuracy Score Without sklearn:", score / float(len(predTree)))

#dot_data = StringIO()
#filename = "Gini.png"
#featureNames = df.columns[0:5]
#targetNames = df["Drug"].unique().tolist()
#out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(Y_train), filled=True,  special_characters=True,rotate=False)  
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100, 200))
#plt.imshow(img,interpolation='nearest')