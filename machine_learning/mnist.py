# Import libraries
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np


st.write("""
# MNIST Digit Recognition Web App
This app recognizes handwritten digits using the MNIST dataset!
***
""")

# Load MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display a sample image
st.subheader('Sample Image')
index = st.slider('Select an image index', 0, x_train.shape[0])
st.image(x_train[index], caption=f"Label: {y_train[index]}", width=150)

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Train the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
st.write(f"Test Accuracy: {test_acc}")

# Make predictions
predictions = model.predict(x_test)
predicted_labels = [tf.argmax(pred).numpy() for pred in predictions]

# Display a random sample of predicted images
st.subheader('Sample Predictions')
num_samples = 5
random_indices = np.random.choice(len(x_test), num_samples)
for index in random_indices:
    st.image(x_test[index], caption=f"Predicted: {predicted_labels[index]}, Actual: {y_test[index]}", width=150)
