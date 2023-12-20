from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

iris = load_iris()
x = iris.data  # x=features
y = iris.target  # y=target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
max_depth = 3
dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(x_train, y_train)
print(dt.predict(x_test))
V = dt.predict(x_test)
result = accuracy_score(y_test, V)  # testing the predicted data with the actual target variable
print("accuracy score :", result)
result1 = classification_report(y_test, V)
print("\n classification report :\n", result1)
plt.figure(figsize=(10,7))
plot_tree(dt,rounded=True,filled=True,feature_names=iris.feature_names,class_names=iris.target_names)
plt.show()
new_data = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)
new_pred = dt.predict(new_data)

pred = iris.target_names[new_pred[0]]
print("Predicted class:", pred)
