import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles", "age"]].values
y = data["Price"].values

# split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create linear regression model
model = LinearRegression()

# fit the model with training data
model.fit(x_train, y_train)

# Find and print the coefficients, intercept, and r squared values.
# Each should be rounded to two decimal places.
coefficients = np.round(model.coef_, 2)
intercept = np.round(model.intercept_, 2)
r_squared = np.round(model.score(x_test, y_test), 2)

print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")

# Loop through the data and print out the predicted prices and the
# actual prices
print("***************")
print("Testing Results")
predictions = model.predict(x_test)
for index in range(len(predictions)):
    print(f"Predicted Price: {np.round(predictions[index], 2)}, Actual Price: {y_test[index]}")

car1_features = np.array([[89000, 10]])  
predicted_price_car1 = model.predict(car1_features)
print(f"Predicted Price for a 10-year-old car with 89,000 miles: ${np.round(predicted_price_car1[0], 2)}")

car2_features = np.array([[150000, 20]]) 
predicted_price_car2 = model.predict(car2_features)
print(f"Predicted Price for a 20-year-old car with 150,000 miles: ${np.round(predicted_price_car2[0], 2)}")