import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.saving import load_model

data = pd.read_csv('kc_house_data (2).csv')


# Separate the data as test and train data
X = data.drop(['price', 'date', 'id'], axis=1)
# X = data[['bedrooms', 'bathrooms', 'sqft_living']]
y = data['price']

# Standardize the data
scale = StandardScaler()   # Initialize the scaler
scale.fit(X)   # Fit it to the data
scaled_X = scale.transform(X)   # Transform the data according to the fitted scaler

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

model = load_model('kc_house_model.h5')

y_pred = model.predict(X_test)[:, 0]

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Root Mean Squared Error:', rmse)

# Plot the predicted values against the actual values using a linear regression model
plt.scatter(y_test, y_pred/y_test)
# Plot a line x = y
plt.plot([0, max(max(y_test), max(y_pred))], [1, 1], color='red')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Predicted Price vs Actual Price')
plt.show()