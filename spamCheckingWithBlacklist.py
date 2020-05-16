# tested on Anaconda, jupyter notebook 6.0.3 
# dataset from https://plg.uwaterloo.ca/~gvcormac/treccorpus07/
import os
import pickle
import email_read_util

DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

# spamfile = open("spamfile.txt", "w")

# Read the labels
with open(LABELS_FILE) as f:
	for line in f:
		line = line.strip()
		label, key = line.split()
		inmail = key.split('/')[-1]
		labels[inmail] = 1 if label.lower() == 'ham' else 0
		# print(inmail, labels[inmail])

# Split corpus into train and test sets
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]
# print(len(X_train))
# print(len(X_test))
'''
for eachline in X_test:
	spamfile.write(eachline + '\n')
spamfile.close()
'''
if not os.path.exists('blacklist.pkl'):
	for filename in X_train:
		path = os.path.join(DATA_DIR, filename)
		if filename in labels:
			# print(filename)
			label = labels[filename]
			stems = email_read_util.load(path)
			if not stems:
				continue
			if label == 1:
				ham_words.update(stems)
			elif label == 0:
				spam_words.update(stems)
			else:
				continue
	blacklist = spam_words - ham_words
	pickle.dump(blacklist, open('blacklist.pkl', 'wb'))
else:
	blacklist = pickle.load(open('blacklist.pkl', 'rb') )

print('Blacklist of {} tokens successfully built/loaded'.format(len(blacklist)))

from nltk.corpus import words
word_set = set(words.words())
word_set.intersection(blacklist)

fp = 0
tp = 0
fn = 0
tn = 0

for filename in X_test:
	path = os.path.join(DATA_DIR, filename)
	if filename in labels:
		# print(filename)
		label = labels[filename]
		stems = email_read_util.load(path)
		if not stems:
			continue
		stems_set = set(stems)
		if stems_set & blacklist:
			if label == 1:
				fp = fp + 1
			else:
				tp = tp + 1
		else:
			if label == 1:
				tn = tn + 1
			else:
				fn = fn + 1

# from IPython.display import HTML, display
conf_matrix = [[tn, fp],[fn, tp]]
# display(HTML('<table><tr>{}</tr></table>'.format('</tr><tr>'.join('<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in conf_matrix))))
print("Confusion Matrix:")
print(conf_matrix)

count = tn + tp + fn + fp
percent_matrix = [["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)], ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]]
# display(HTML('<table><tr>{}</tr></table>'.format('</tr><tr>'.join('<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in percent_matrix))))
print("Confusion Matrix as percentage of total count {} of emails:", count)
print(percent_matrix)

print("Classification accuracy: {}".format("{:.1%}".format((tp+tn)/count)))
