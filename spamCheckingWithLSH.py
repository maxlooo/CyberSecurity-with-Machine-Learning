# tested on Anaconda, jupyter notebook 6.0.3 
# dataset from https://plg.uwaterloo.ca/~gvcormac/treccorpus07/
import os
import pickle
import email_read_util
from datasketch import MinHash, MinHashLSH

DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}

# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Split corpus into train and test sets
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

# Extract only spam files for inserting into the LSH matcher
spam_files = [x for x in X_train if labels[x] == 0]

# Initialize MinHashLSH matcher with a Jaccard 
# threshold of 0.5 and 128 MinHash permutation functions
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# Populate the LSH matcher with training spam MinHashes
for filename in spam_files:
    minhash = MinHash(num_perm=128)
    filepath = os.path.join(DATA_DIR, filename)
    stems = email_read_util.load(filepath)
    if len(stems) < 2: continue
    for s in stems:
        minhash.update(s.encode('utf-8'))
    lsh.insert(filename, minhash)

def lsh_predict_label(stems):
    '''
    Queries the LSH matcher and returns:
        0 if predicted spam
        1 if predicted ham
       -1 if parsing error
    '''
    minhash = MinHash(num_perm=128)
    if len(stems) < 2:
        return -1
    for s in stems:
        minhash.update(s.encode('utf-8'))
    matches = lsh.query(minhash)
    if matches:
        return 0
    else:
        return 1

fp = 0
tp = 0
fn = 0
tn = 0

for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = email_read_util.load(path)
        if not stems:
            continue
        pred = lsh_predict_label(stems)
        if pred == -1:
            continue
        elif pred == 0:
            if label == 1:
                fp += 1
            else:
                tp += 1
        elif pred == 1:
            if label == 1:
                tn += 1
            else:
                fn += 1

conf_matrix = [[tn, fp],
               [fn, tp]]
import numpy as np
cm = np.array(conf_matrix)
print("Confusion Matrix:")
print(cm)
# print(cm[:1,:])
# print(cm[1:,:])

count = tn + tp + fn + fp
percent_matrix = [["{:.1%}".format(tn/count), "{:.1%}".format(fp/count)],
                  ["{:.1%}".format(fn/count), "{:.1%}".format(tp/count)]]
print("Confusion Matrix in Percentage of Total Count: ")
# print(percent_matrix)
print(percent_matrix[:1])
print(percent_matrix[1:])

print("Classification accuracy: {}".format("{:.1%}".format((tp+tn)/count)))
