# elliptic envelope fitting to detect outliers from 
# normal distribution and low dimensional datasets
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
# %matplotlib inline

pdf = PdfPages('AnomalyDetectionVariousMethods.pdf')

plt.rcParams["figure.figsize"] = (8,6)

def output(x, outlier_ratio):
	scores_pred = classifier.decision_function(x)
	threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_ratio)
	spacing = np.linspace(-11,11,1000)
	xx, yy = np.meshgrid(spacing, spacing)
	Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	return xx, yy, Z, threshold

num_dimensions = 2
num_samples = 1000
outlier_ratio = 0.01
num_inliers = int(num_samples * (1-outlier_ratio))
num_outliers = num_samples - num_inliers

# Generate the normally-distributed inliers
x = np.random.randn(num_inliers, num_dimensions)

# Add outliers sampled from a random uniform distribution
x_rand = np.random.uniform(low=-10, high=10, size=(num_outliers, num_dimensions))
x = np.r_[x, x_rand]

# Generate labels, 1 for inliers and -1 for outliers
labels = np.ones(num_samples, dtype=int)
labels[-num_outliers:] = -1

# plt.figure()
inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')
plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')

## Applying sklearn.covariance.EllipticEnvelope

classifier = EllipticEnvelope(contamination=outlier_ratio)
classifier.fit(x)
y_pred = classifier.predict(x)
num_errors = sum(y_pred != labels)
print('Number of errors fitting Elliptic Envelope to Gaussian distribution: {}'.format(num_errors))

xx, yy, Z, threshold = output(x, outlier_ratio)

# plt.figure()
inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')

plt.contour(xx, yy, Z, levels=[threshold],linewidths=5, colors='gray')
plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Greys_r)
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='gray')
plt.xlim(-11,11)
plt.ylim(-11,11)

plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')
plt.clf()

## Example application on non-Gaussian distribution

x_0 = np.random.randn(num_inliers//3, num_dimensions) - 3
x_1 = np.random.randn(num_inliers//3, num_dimensions)
x_2 = np.random.randn(num_inliers//3, num_dimensions) + 4

# Add outliers sampled from a random uniform distribution
x_rand = np.random.uniform(low=-10, high=10, size=(num_outliers, num_dimensions))
x = np.r_[x_0, x_1, x_2, x_rand]

# Generate labels, 1 for inliers and -1 for outliers
labels = np.ones(num_samples, dtype=int)
labels[-num_outliers:] = -1

# plt.figure()
inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')
plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')

classifier = EllipticEnvelope(contamination=outlier_ratio)
classifier.fit(x)
y_pred = classifier.predict(x)
num_errors = sum(y_pred != labels)
print('Number of errors fitting Elliptic Envelope to non-Gaussian distribution: {}'.format(num_errors))

xx, yy, Z, threshold = output(x, outlier_ratio)

# plt.figure()
inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')

plt.contour(xx, yy, Z, levels=[threshold],linewidths=5, colors='gray')
plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Greys_r)
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='gray')

plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')
plt.clf()

# one class svm for novelty detection in non-Gaussian or 
# multimodal distributions and high dimensional datasets.
# datasets should be cleaned before training.  
x_0 = np.random.randn(num_inliers//3, num_dimensions)
x_1 = np.random.randn(num_inliers//3, num_dimensions) - 7
x_2 = np.random.randn(num_inliers//3, num_dimensions) - 7
x_rand = np.random.uniform(low=-10, high=10, size=(num_outliers, num_dimensions))
# print(np.shape(x_rand))

# Add outliers sampled from a random uniform distribution
x = np.r_[x_0, x_1, x_2, x_rand]

# Generate labels, 1 for inliers and -1 for outliers
labels = np.ones(num_samples, dtype=int)
labels[-num_outliers:] = -1

## Applying sklearn.svm.OneClassSVM

classifier = svm.OneClassSVM(nu=0.99 * outlier_ratio + 0.01, kernel="rbf", gamma=0.1)
classifier.fit(x)
y_pred = classifier.predict(x)
num_errors = sum(y_pred != labels)
print('Number of errors for strongly bimodal dataset that is uncleaned: {}'.format(num_errors))

xx, yy, Z, threshold = output(x, outlier_ratio)

inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')

plt.contour(xx, yy, Z, levels=[threshold],linewidths=5, colors='gray')
plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Greys_r)
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='gray')

plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')
plt.clf()

# dataset is cleaned of outliers
x = np.r_[x_0, x_1, x_2]
classifier = svm.OneClassSVM(nu=0.99 * outlier_ratio + 0.01, kernel="rbf", gamma=0.1)
classifier.fit(x)

xx, yy, Z, threshold = output(x, outlier_ratio)

x = np.r_[x, x_rand]

inlier_plot = plt.plot(x[:num_inliers,0], x[:num_inliers,1], 'go', label='inliers')
outlier_plot = plt.plot(x[-num_outliers:,0], x[-num_outliers:,1], 'ko', label='outliers')

plt.contour(xx, yy, Z, levels=[threshold],linewidths=5, colors='gray')
plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Greys_r)
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='gray')

plt.xlim(-11,11)
plt.ylim(-11,11)
plt.legend(numpoints=1)
# plt.show()
plt.savefig(pdf, format='pdf')
plt.clf()

pdf.close()
