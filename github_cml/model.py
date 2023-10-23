# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model

# Define a function to plot confusion matrix for classification results
def plot_classification_results(test_labels, predictions, classes):
    """
    Plots confusion matrix for classification results.
    """
    # Calculate confusion matrix and create a ConfusionMatrixDisplay object
    cm = ConfusionMatrixDisplay.from_predictions(y_test_cat.argmax(axis=1), predictions.argmax(axis=1))
    return cm

# Define a function to calculate accuracy_score between y_test and y_preds
def accuracy(y_test, y_pred):
    """
    Calculates accuracy_score between y_test and y_preds.
    """
    return accuracy_score(y_test, y_pred)

# Define a function to calculate classification_report between y_test and y_preds
def classification(y_test, y_pred):
    """
    Calculates classification_report between y_test and y_preds.
    """
    return classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Define a function to plot the loss curve
def loss_plot(lo):
    """
    Plots the loss curve using the data provided.
    """
    return lo.plot()

# Define a function to save the model architecture as an image file
def plo_model(mo):
    """
    Saves the model architecture as an image file.
    """
    plot_model(mo, to_file='model.png', show_shapes=True)

# Load MNIST dataset and preprocess it
(x_train, y_train), (x_test, y_test) = mnist.load_data()
classes = list(range(10))
y_train_cat = to_categorical(y_train).astype(int)
y_test_cat = to_categorical(y_test).astype(int)
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Build a convolutional neural network model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', input_shape=[28, 28, 1], activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to prevent overfitting
early = EarlyStopping(monitor='val_accuracy', mode='max', patience=2)

# Train the model and validate it on the test set
model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=20, callbacks=[early])

# Get the loss data for plotting
loss = pd.DataFrame(model.history.history)

# Make predictions using the test data
predictions = (model.predict(x_test) > 0.5).astype('int32')

# Plot the confusion matrix
cm = plot_classification_results(y_test_cat, predictions, classes)
# Plot the loss curve
los = loss_plot(loss)
# Save the model architecture as an image file
plo = plo_model(model)

# Calculate accuracy score
acc = accuracy(y_test_cat, predictions)

# Calculate classification report
clas = classification(y_test_cat, predictions)

# Print accuracy score and write metrics to a file
print(f'\naccuracy_score = {acc}')
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\naccuracy_score = {acc}, classification_report = {clas}, loss_plot = {los}, plot_model = {plo}')
