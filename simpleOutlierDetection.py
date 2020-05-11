import numpy as np

# input data series
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 99]

# calculate median absolute deviation
mad = np.median(np.abs(x - np.median(x)))

# MAD of x is 1.5
print('MAD: {}'.format(mad))

for i in x:
	if (i - np.median(x)) > mad * 2:
		print("{} is outlier".format(i))

[print("{} is outlier".format(i)) for i in x if (i - np.median(x)) > mad * 2]
