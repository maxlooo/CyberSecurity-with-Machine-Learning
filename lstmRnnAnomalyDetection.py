# This python program is run from Google Colab with Jupyter
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

from google.colab import files

epochs = 5
batch_size = 50
# Each training data point will be length 100-1,
# since the last value in each sequence is the label
sequence_length = 50

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = '19AcjLK1Vc_g_Oo_ceRj_UNp8TTgDgPZ6'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('cpu-full-b.csv')
# print('Downloaded content "{}"'.format(downloaded.GetContentString()))

def normalize(result):
	result_mean = result.mean()
	result_std = result.std()
	result -= result_mean
	result /= result_std
	return result, result_mean

global_start_time = time.time()

print('Loading data... ')
data_b = pd.read_csv('cpu-full-b.csv', parse_dates=[0], infer_datetime_format=True)
data = data_b['cpu'].to_numpy()  # as_matrix() doesn't work

# train on first 600 samples and test on last 260 samples, 
# 200 of which overlaps with training (test set has anomaly)
train_start, train_end, test_start, test_end = 0, 550, 400, 660
print("Length of Data", len(data))

# training data
print("Creating training data...")

result = []
for index in range(train_start, train_end - sequence_length):
	result.append(data[index: index + sequence_length])
result = np.array(result)
result, result_mean = normalize(result)

print("Training data shape  : ", result.shape)

train = result[train_start:train_end, :]
np.random.shuffle(train)
X_train = train[:, :-1]
y_train = train[:, -1]

# test data
print("Creating test data...")

result = []
for index in range(test_start, test_end - sequence_length):
	result.append(data[index: index + sequence_length])
result = np.array(result)
result, result_mean = normalize(result)

print("Test data shape  : {}".format(result.shape))

X_test = result[:, :-1]
y_test = result[:, -1]

print("Shape X_train", np.shape(X_train))
print("Shape X_test", np.shape(X_test))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()

# First LSTM layer defining the input sequence length
model.add(LSTM(input_shape=(sequence_length-1, 1), units=64, return_sequences=True))
model.add(Dropout(0.2))
# Second LSTM layer with 128 units
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer with 100 units
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# Densely-connected output layer with the linear activation function
model.add(Dense(units=1))
model.add(Activation('linear'))
model.summary()
plot_model(model, to_file='lstmRnnGraph1.pdf')
files.download("lstmRnnGraph1.pdf") 

model.compile(loss='mean_squared_error', optimizer='adam')

print("Training...")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
print("Predicting...")
predicted = model.predict(X_test)
print("Reshaping predicted")
predicted = np.reshape(predicted, (predicted.size,))

plt.figure(figsize=(20,8))
plt.plot(y_test[:len(y_test)], 'b', label='Observed')
plt.plot(predicted[:len(y_test)], 'g', label='Predicted')
plt.plot(((y_test - predicted) ** 2), 'r', label='Root-mean-square deviation')
plt.legend()
# plt.show()
plt.savefig("lstmRnnGraph2.pdf")
files.download("lstmRnnGraph2.pdf") 
print('Training duration:{}'.format(time.time() - global_start_time))
