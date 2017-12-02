import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

# Splicing independent from dependent variables
# The Level column is related to Position column, that is the why Position was not considered
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Polynomial Regression model
polynomFeat = PolynomialFeatures(degree=4)
X_poly = polynomFeat.fit_transform(X)
linearRegrPoly = LinearRegression()
linearRegrPoly.fit(X_poly, y)

# Linear Regression model
linearRegr = LinearRegression()
linearRegr.fit(X, y)
linearRegr.predict(X)

# Plotting graphs of Linear regression
plt.scatter(X, y, color="blue")
plt.plot(X, linearRegr.predict(X), color="green")
plt.xlabel("Salary")
plt.ylabel("Level")
plt.show()

# Ploting the Polynomial Regression Model
plt.scatter(X, y, color="blue")
plt.plot(X, linearRegrPoly.predict(polynomFeat.fit_transform(X)),
         color="green")
plt.xlabel("Salary")
plt.ylabel("Level")
plt.show()

# Predicting new results with Linear Regression
print(linearRegr.predict(6.5))
print(linearRegr.predict(9.5))
print(linearRegr.predict(2.5))

print("\n")

# Predictions using the Polynomial Model
print(linearRegrPoly.predict(polynomFeat.fit_transform(6.5)))
print(linearRegrPoly.predict(polynomFeat.fit_transform(9.5)))
print(linearRegrPoly.predict(polynomFeat.fit_transform(2.5)))
