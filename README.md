# LGMVIP---DataScience
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#creating columns names list because in our data file there are not appropriate names 
columns=['Sepal Length','Sepal Width','Petel Length','Petel Width','Class_labels'] 
# loading data
df = pd.read_csv("C:\\Users\\TAMANNA\\Downloads\\iris.data", names=columns)
df.head()

df.describe()

sns.pairplot(df, hue='Class_labels')

#from this visualization we can see that:
:- Iris-setosa is well separated from the other two flowers
:- Iris-virginica is the longest flower and iris-setosa is the shortest

#separate features and target:
data = df.values
X = data[:,0:4]
Y = data[:,4]

# calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))]) 
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0,1)
X_axis = np.arange(len(columns)-1)
width = 0.25

#Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label="Setosa")
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label="Versicolour")
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label="Virginica")
plt.xticks(X_axis, columns[0:4])
plt.xlabel("Features")
plt.ylabel("Value in cm")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

#model training
# split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, Y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)
