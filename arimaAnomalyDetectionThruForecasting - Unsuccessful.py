# using python 3.7.6 with anaconda
# pip3 install pyflux==0.3.2 only, anaconda cannot find pyflux packages
# pyflux version is outdated
# datasets can be obtained from 
# https://github.com/oreilly-mlsec/book-resources/tree/master/chapter3/datasets/cpu-utilization
import os
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
# %matplotlib inline

dataset_root = 'datasets'

data_a = pd.read_csv(os.path.join(dataset_root, 'cpu-full-a.csv'), parse_dates=[0], infer_datetime_format=True)
data_train_a = pd.read_csv(os.path.join(dataset_root, 'cpu-train-a.csv'), parse_dates=[0], infer_datetime_format=True)
data_test_a = pd.read_csv(os.path.join(dataset_root, 'cpu-test-a.csv'), parse_dates=[0], infer_datetime_format=True)

data_b = pd.read_csv(os.path.join(dataset_root, 'cpu-full-b.csv'), parse_dates=[0], infer_datetime_format=True)
data_train_b = pd.read_csv(os.path.join(dataset_root, 'cpu-train-b.csv'), parse_dates=[0], infer_datetime_format=True)
data_test_b = pd.read_csv(os.path.join(dataset_root, 'cpu-test-b.csv'), parse_dates=[0], infer_datetime_format=True)

with PdfPages('moreArimaPlots.pdf') as pdf:
	a = plt.figure(figsize=(20,8))
	plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black')
	plt.ylabel('CPU %')
	plt.title('CPU Utilization')
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	b = plt.figure(figsize=(20,8))
	plt.plot(data_a['datetime'], data_a['cpu'], color='black')
	plt.ylabel('CPU %')
	plt.title('CPU Utilization')
	# axvspan to highlight part of full data that's not used for training
	plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,1,42)), xmax=pd.Timestamp(datetime(2017,1,28,2,41)), color='#d4d4d4')
	pdf.savefig()  
	plt.close()

	c = plt.figure(figsize=(20,8))
	plt.plot(data_train_b['datetime'], data_train_b['cpu'], color='black')
	plt.ylabel('CPU %')
	plt.title('CPU Utilization')
	pdf.savefig()  
	plt.close()

	d = plt.figure(figsize=(20,8))
	plt.plot(data_b['datetime'], data_b['cpu'], color='black')
	plt.ylabel('CPU %')
	plt.title('CPU Utilization')
	plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,4,42)), xmax=pd.Timestamp(datetime(2017,1,28,5,41)), color='#d4d4d4')
	pdf.savefig()  # last savefig here, ARIMA plots cannot use savefig
	plt.close()

	model_a = pf.ARIMA(data=data_train_a, ar=11, ma=11, integ=0, target='cpu')
	x = model_a.fit("M-H")
	model_a.plot_fit(figsize=(20,8))
	# plot_predict fails, AttributeError: module 'pandas.tseries' has no attribute 'index'
	# problem likely arises from outdated pyflux version==0.3.2
	# unable to install later versions of pyflux due to use of anaconda
	model_a.plot_predict(h=60,past_values=100,figsize=(20,8))
	
	y = model_a.plot_predict_is(h=60, figsize=(20,8))
	
	model_b = pf.ARIMA(data=data_train_b, ar=11, ma=11, integ=0, target='cpu')
	z = model_b.fit("M-H")
	model_b.plot_predict(h=60,past_values=100,figsize=(20,8))
	
#pp = PdfPages('allArimaPlots.pdf')
#pp.savefig(a)
#pp.savefig(b)
#pp.savefig(c)
#pp.savefig(d)
#pp.savefig(x)
#pp.savefig(y)
#pp.savefig(z)
#pp.close()
