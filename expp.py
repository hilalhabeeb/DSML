import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Salary_Data.csv')
x = data['YearsExperience'].values.reshape(-1,1)
y= data['Salary'].values
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=27)
clf = LinearRegression()
clf.fit(x_train,y_train)
print(clf.predict(x_test))
v=clf.predict(x_test)
result=r2_score(y_test, v)
print("R squared:", result)


# to plot the graph
plt.scatter(x_test, y_test, color='black',label='Data Points')
plt.plot(x_test,v, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()
plt.show()