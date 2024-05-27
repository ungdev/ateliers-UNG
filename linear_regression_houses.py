import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('kc_house_data.csv')


# Separate the data as test and train data
X = data.drop(['price'], axis=1)
# X = data[['bedrooms', 'bathrooms', 'sqft_living']]
y = data['price']

# Standardize the data
scale = StandardScaler()   # Initialize the scaler
scale.fit(X)   # Fit it to the data
scaled_X = scale.transform(X)   # Transform the data according to the fitted scaler

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Instantiate a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Root Mean Squared Error:', rmse)

# Plot the predicted values against the actual values using a linear regression model
plt.scatter(y_pred, y_test)
# Plot a line x = y
plt.plot([0, max(*y_test, *y_pred)], [0, max(*y_test, *y_pred)], color='red')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Predicted Price vs Actual Price')
plt.show()


