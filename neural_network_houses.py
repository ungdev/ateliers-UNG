# Libraries import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ELU
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('kc_house_data (2).csv')

X = data.drop(['price', 'id', 'date'], axis=1)   # We use every feature, except the price we're trying to predict
y = data['price']

# Standardize the data
scale = StandardScaler()   # Initialize the scaler
scale.fit(X)   # Fit it to the data
scaled_X = scale.transform(X)   # Transform the data according to the fitted scaler

# Separate the data as test and train data
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()   # Initialize the structure of the neural network
model.add(Dense(10, input_dim=X_train.shape[1]))   # Add a layer of 10 neurons, the input layer is implicit with X_train.shape[1] neurons
model.add(LeakyReLU(alpha=0.1))   # Tell the model that those first 10 neurons are of activation function LeakyReLU. alpha = slope for negative values (see LeakyReLU graph)
model.add(Dense(32))   # Add a layer of 32 neurons, input shape is inferred from last layer
model.add(ELU(alpha=1.0))   # Add the ELU activation function for those 32 neurons
model.add(Dense(64))   # Add a layer of 64 neurons
model.add(LeakyReLU(alpha=0.1))   # Add the LeakyReLU activation function for those 64 neurons
model.add(Dense(1, activation='linear'))   # Use a linear activation function (= no activation) for output layer. 1 neuron since we want 1 value

optimizer = Adam(learning_rate=0.003)   # Optimizer defines the process to diminish the cost function. Here, we use the Adam optimizer, an already existing algorithm

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict house prices
y_pred = model.predict(X_test)[:, 0]

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Root Mean Squared Error:', rmse)

# Plot the predicted values against the actual values using a linear regression model
plt.scatter(y_pred, y_test)
# Plot a line x = y
plt.plot([0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))], color='red')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Predicted Price vs Actual Price')
plt.show()