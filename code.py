#simple linear regression model salary prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("salary.csv")
#seperating the dataset as input and output
X=data.iloc[:,:-1]
Y=data.iloc[:,1]
print(X.head(5))
print(Y.head(5))

#test_train_split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=45)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

lr.predict(x_test)

plt.scatter(x_train,y_train)
plt.plot(x_train,lr.predict(x_train))
plt.title("salary_pridiction_based_on_exprinece")
plt.xlabel("year")
plt.ylabel("salary")
plt.show()
