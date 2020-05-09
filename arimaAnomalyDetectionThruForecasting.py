# Using Google Colab's Jupyter for python and pyflux
# datasets can be obtained from 
# https://github.com/oreilly-mlsec/book-resources/tree/master/chapter3/datasets/cpu-utilization
import os
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
%matplotlib inline

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
file_id_a = '1xB8UuA8k_n88gkuZQVl6ZCInfRpdBwQH'
downloaded_a = drive.CreateFile({'id': file_id_a})
downloaded_a.GetContentFile('cpu-full-a.csv')
file_id_train_a = '1RN_qFnCsaD25gwFJyTkuFbH67EvUEkzx'
downloaded_train_a = drive.CreateFile({'id': file_id_train_a})
downloaded_train_a.GetContentFile('cpu-train-a.csv')
file_id_test_a = '1DsR37-wZCCzdU7tqvRQ9OZoeSmxlKG4N'
downloaded_test_a = drive.CreateFile({'id': file_id_test_a})
downloaded_test_a.GetContentFile('cpu-test-a.csv')
file_id_b = '19AcjLK1Vc_g_Oo_ceRj_UNp8TTgDgPZ6'
downloaded_b = drive.CreateFile({'id': file_id_b})
downloaded_b.GetContentFile('cpu-full-b.csv')
file_id_train_b = '1tHqM5qRzVuUuGmR6KOIne6CIJsR1LJGe'
downloaded_b_train_b = drive.CreateFile({'id': file_id_train_b})
downloaded_b_train_b.GetContentFile('cpu-train-b.csv')
file_id_test_b = '12yA4_8lrxUcL2qDEH2GR6_Zc5V23u25U'
downloaded_b_test_b = drive.CreateFile({'id': file_id_test_b})
downloaded_b_test_b.GetContentFile('cpu-test-b.csv')

data_a = pd.read_csv('cpu-full-a.csv', parse_dates=[0], infer_datetime_format=True)
data_train_a = pd.read_csv('cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True)
data_test_a = pd.read_csv('cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True)

data_b = pd.read_csv('cpu-full-b.csv', parse_dates=[0], infer_datetime_format=True)
data_train_b = pd.read_csv('cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True)
data_test_b = pd.read_csv('cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True)

plt.figure(figsize=(20,8))
plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')

plt.figure(figsize=(20,8))
plt.plot(data_a['datetime'], data_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')
# axvspan to highlight part of full data that's not used for training
plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,1,42)), xmax=pd.Timestamp(datetime(2017,1,28,2,41)), color='#d4d4d4')

plt.figure(figsize=(20,8))
plt.plot(data_train_b['datetime'], data_train_b['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')

plt.figure(figsize=(20,8))
plt.plot(data_b['datetime'], data_b['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')
plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,4,42)), xmax=pd.Timestamp(datetime(2017,1,28,5,41)), color='#d4d4d4')

model_a = pf.ARIMA(data=data_train_a, ar=11, ma=11, integ=0, target='cpu')
x = model_a.fit("M-H")
model_a.plot_fit(figsize=(20,8))
model_a.plot_predict(h=60,past_values=100,figsize=(20,8))
	
model_a.plot_predict_is(h=60, figsize=(20,8))
	
model_b = pf.ARIMA(data=data_train_b, ar=11, ma=11, integ=0, target='cpu')
y = model_b.fit("M-H")
model_b.plot_predict(h=60,past_values=100,figsize=(20,8))
