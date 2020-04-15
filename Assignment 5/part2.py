
# TDT4171 Assignment 5
# @author Anastasia Lindbäck
#
# Part 2
#
import pickle
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
print("Loading data...")
data = pickle.load(open("keras-data.pickle", "rb"))

x_train, y_train = data.get('x_train'), data.get('y_train')
x_test, y_test = data.get('x_test'), data.get('y_test')

# Pad sequences
print("Padding sequences...")

max_length = int(data.get('max_length')/8)

X_train = pad_sequences(sequences=x_train, maxlen=max_length)
X_test = pad_sequences(sequences=x_test, maxlen=max_length)

# Defining model
print("Defining model...")

vocab_size = data.get('vocab_size')
embedding_size = 32
hidden_nodes = 16

model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length)) # Embedding layer
model.add(LSTM(hidden_nodes)) # LSTM layer
model.add(Dense(units=1, activation='sigmoid')) # Dense layer

# Compiling model
print("Compiling model..")
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Fitting...")

epochs = 3

model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate model
test_loss, accuracy = model.evaluate(x=X_test, y=y_test)

print("\n\n")
print("Loss of LSTM model ", test_loss)
print("Accuracy of LSTM model ", accuracy)
