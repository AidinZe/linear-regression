#aidin_zehtab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

fileName = input("please inter csv file name:")
# Load CSV and columns
my_data = pd.read_csv(fileName)

colmunsName1=input("inter colmunsName1 or default 0:")
if (colmunsName1 == '0'):
    data1 = my_data.columns[1]
    data2 = my_data.columns[2]
else:
    colmunsName2=input("inter colmunsName2:")
    data1 = colmunsName1
    data2 = colmunsName2

X = my_data[data1]
Y = my_data[data2]

X = X.values.reshape(len(X),1)
Y = Y.values.reshape(len(Y),1)

# Split the data into training/testing sets
X_train = X[:-500]
X_test = X[-500:]

# Split the targets into training/testing sets
y_train = Y[:-500]
y_test = Y[-500:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred))
#Plot outputs

plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, diabetes_y_pred, color='red', linewidth=3)

plt.xlabel(data1)
plt.ylabel(data2)
plt.title(fileName)

plt.xticks(())
plt.yticks(())


plt.grid(True)
plt.show()
