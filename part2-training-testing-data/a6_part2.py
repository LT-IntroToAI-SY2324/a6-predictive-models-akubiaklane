import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1,1)

# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values. 
coefficient = round(float(model.coef_), 2)
bias = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)

# Each should be a float and rounded to two decimal places. 


# Print out the linear equation and r squared value:
print(f"The equation is y = {coefficient}x + {bias}")
print(f"r squared value is {r_squared}")

'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1, 1)

# get the predicted y values for the xtest values - returns an array of the results
predictions = model.predict(xtest)

# round the value in the np array to 2 decimal places
predictions = np.around(predictions, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")

for index in range(len(xtest)):
    actual_value = ytest[index]
    prediction = predictions[index]
    coordinate = xtest[index]

    print("x value:", float(coordinate), "Predicted y value:", prediction, "Actual y value:", actual_value)
    
'''
**********CREATE A VISUAL OF THE RESULTS**********
'''

plt.figure(figsize=(6, 4))

plt.scatter(xtrain, ytrain, c="blue", label="Training Data")
plt.scatter(xtest, ytest, c="red", label="Testing Data")
plt.scatter(xtest, predictions, c="purple", label="Predictions")

plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")
plt.plot(x, coefficient*x + bias, c="r", label="Line of Best Fit")

plt.legend()
plt.show()