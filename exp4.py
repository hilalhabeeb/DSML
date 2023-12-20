from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
iris = load_iris()
x=iris.data  # x=features
y=iris.target  # y=target
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=48)
# x_train = 80% training data of features  x_test =20% testing data of features
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
print(knn.predict(x_test))
V=knn.predict(x_test)
result=accuracy_score (y_test,V) # testing the predicted data with the actual target variable
print("accuracy score :", result)

unseen_data = [[5.1, 3.5, 1.4, 0.2]]  # Example features (modify with your own data)

predicted_category = knn.predict(unseen_data)
predicted_target_name = iris.target_names[predicted_category]
print("Predicted category for the unseen data:", predicted_target_name)