import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st

# Load IMDB dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True,split='train',shuffle_files=True)
train = imdb['train']
test = imdb['test']

# Process the dataset
train_sequences = []
train_labels = []
test_sequences = []
test_labels = []
for s, l in train:
    train_sequences.append(s.numpy().decode('utf-8'))
    train_labels.append(l.numpy())
for s, l in train:
    test_sequences.append(s.numpy().decode('utf-8'))
    test_labels.append(l.numpy())

train_l = np.array(train_labels)
test_l = np.array(test_labels)

vocab_size = 1000
max_len = 120
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(train_sequences)
word_index = tokenizer.word_index
train_sequence = tokenizer.texts_to_sequences(train_sequences)
train_pad = pad_sequences(train_sequence, truncating='post', maxlen=max_len)
test_sequence = tokenizer.texts_to_sequences(test_sequences)
test_pad = pad_sequences(test_sequence, truncating='post', maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_shape=[max_len]))
model.add(LSTM(100))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_pad, train_l, validation_data=[test_pad, test_l], epochs=10)

# Evaluate the model
metrics = pd.DataFrame(model.history.history)

# Display loss and accuracy charts
st.subheader('Loss')
st.line_chart(metrics[['loss', 'val_loss']])

st.subheader('Accuracy')
st.line_chart(metrics[['accuracy', 'val_accuracy']])

# Display a sample image
st.subheader('Sample Image')
index = st.slider('Select an image index', 0, x_train.shape[0])
st.image(x_train[index], caption=f"Label: {y_train[index]}", width=150)

# Make predictions
predictions = (model.predict(test_pad) > 0.5).astype(int)

# Display confusion matrix
cm = ConfusionMatrixDisplay.from_estimator(model, test_pad, test_l)
st.subheader('Confusion Matrix')
st.write(cm.confusion_matrix)
