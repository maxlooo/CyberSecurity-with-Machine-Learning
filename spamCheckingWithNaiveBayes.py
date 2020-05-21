# tested on Anaconda, jupyter notebook 6.0.3 
# dataset from https://plg.uwaterloo.ca/~gvcormac/treccorpus07/
import os
import email_read_util

DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        inmail = key.split('/')[-1]
        labels[inmail] = 1 if label.lower() == 'ham' else 0

def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        filepath = os.path.join(DATA_DIR, filename)
        email_str = email_read_util.extract_email_text(filepath)
        X.append(email_str)
        y.append(labels[filename])
    return X, y

X, y = read_email_files()

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = \
    (1-TRAINING_SET_RATIO), random_state=2)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Initialize the classifier and make label predictions
mnb = MultinomialNB()
mnb.fit(X_train_vector, y_train)
y_pred = mnb.predict(X_test_vector)

# Print results
print("Results using CountVectorizer for feature extraction:")
print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))

print('\n', "Results using TF/IDF Vectorizer for feature extraction:")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
X_train_vector2 = vectorizer2.fit_transform(X_train)
X_test_vector2 = vectorizer2.transform(X_test)

mnb2 = MultinomialNB()
mnb2.fit(X_train_vector2, y_train)
y_pred2 = mnb2.predict(X_test_vector2)
print(classification_report(y_test, y_pred2, target_names=['Spam', 'Ham']))
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred2)))
