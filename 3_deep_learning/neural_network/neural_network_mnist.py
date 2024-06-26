from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ReLU
from keras.optimizers import Adam
import numpy as np
from keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)

# Flatten the data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])[:1000]
y_train = y_train[:1000]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# y_train = np.array([np.eye(10)[i] for i in y_train])
# y_test = np.array([[0] * i + [1] + [0] * (9 - i) for i in y_test])


# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1]))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(10, activation='sigmoid'))

optimizer = Adam(learning_rate=0.003)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save('mnist.h5')
