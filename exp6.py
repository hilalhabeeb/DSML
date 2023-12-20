from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
x=iris.data  # x=features
y=iris.target  # y=target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=48)
nb=GaussianNB()
nb.fit(x_train,y_train)
print(nb.predict(x_test))
V=nb.predict(x_test)
result=accuracy_score (y_test,V) # testing the predicted data with the actual target variable
print("accuracy score :", result)