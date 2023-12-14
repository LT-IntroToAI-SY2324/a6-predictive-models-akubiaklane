import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)
# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler().fit(x)
# Step 3: Transform the data
x = scaler.transform(x)
# Step 4: Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y)
# Step 5: Fit the data
scaler.fit(data)
# Step 6: Create a LogsiticRegression object and fit the data
logistic=linear_model.LogisticRegression()
logistic.fit(x_train,y_train)
# Step 7: Print the score to see the accuracy of the model
print(logistic.score(x_test,y_test))
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
print("real Y values:")
print(y_test)
print("predicted Y values:")
print(logistic.predict(x_test))
print(len(y_test))
female_thirtyfour = [["34", "56000", "1"]]
female_thirtyfour=scaler.transform(female_thirtyfour)
my_prediction=logistic.predict(female_thirtyfour)
print(my_prediction) 