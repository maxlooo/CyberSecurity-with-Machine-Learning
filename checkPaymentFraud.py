'''

Payment Data is obtained from:
https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter2/datasets/payment_fraud.csv

'''
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)


def main():
    # import dataset
    dataframe = pandas.read_csv("./PaymentData.csv", verbose=True)
    print("5 lines of data from PaymentData: \n{}".format(dataframe.sample(5)))
        
    # change payment method from categorical variable to numeric
    dataframe.paymentMethod.replace(['paypal', 'creditcard', 'storecredit'], [0,1,2], inplace=True)
    print("\n5 lines of data after changing categorical variable to numeric: \n{}".format(dataframe.sample(5)))
    
    # create the X axis
    X_axis = dataframe.drop('label', axis=1)
    
    # create the y axis
    y_axis = dataframe['label']
    
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_axis, y_axis, test_size=.2, shuffle=True)
    print("\nShape of the X_train is: {}".format(X_train.shape))
    print("Shape of the X_test is: {}".format(X_test.shape))
    print("\nShape of the y_train is: {}".format(y_train.shape))
    print("Shape of the y_test is: {}".format(y_test.shape))

    # Fit training data to logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    print("\nClassifier aftering being fitted: \n{}".format(classifier))

    # Predict y_test data
    y_predict = classifier.predict(X_test)

    # Test the accuracy
    print("\nAccuracy Score: {}".format(accuracy_score(y_predict, y_test)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, y_predict)))
    print("Classification report on your prediction \n{}".format(classification_report(y_test,y_predict)))


    # Predict based on new transaction
    print("Enter a new transaction with 5 feature values separated by commas")
    transaction = input("Example: 1,4,4,1,0: ")
    transaction = transaction.split(',')
    transaction = [int(i) for i in transaction]
    print("Transaction Entered: {}".format(transaction))
    if classifier.predict([transaction]) == 1:
    	print("Transaction is fraudulent!")
    else:
        print("Transaction is OK")
    print("Probability score: {}".format(classifier.predict_proba([transaction])))

if __name__ == '__main__':
    main()
