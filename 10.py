import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

california_housing = fetch_california_housing()
df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
df['Target'] = california_housing.target
x = df.drop('Target', axis=1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")



# from sklearn.datasets import fetch_california_housing
# from sklearn.metrics import  mean_squared_error,r2_score
# from sklearn.linear_model import  LinearRegression
# from sklearn.model_selection import train_test_split
#
# ch=fetch_california_housing()
# x=ch.data
# y=ch.target
# x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=46)
# mr=LinearRegression()
# mr.fit(x_train,y_train)
# v=mr.predict(x_test)
# mean_squared_error=mean_squared_error(y_test,v)
# r2=r2_score(y_test,v)
# print(mean_squared_error)
# print(r2)