import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading the dataset 
credit_card_data = pd.read_csv("D:\College Work\AI\Credit Card Fraud Detection\creditcard.csv")

#Print first 5 rows
print(credit_card_data.head())
print(credit_card_data.tail())

#Dataset information
print(credit_card_data.info())

#Checking the missing values in each column
print(credit_card_data.isnull().sum())

#Distrubution of legit transaction and fradulant transection
print(credit_card_data['Class'].value_counts)

#0----> Normal Transection
#1----> Fraudulant Transection

#Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)

#Statistical measure of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())

#Compare the values for both transactions
print(credit_card_data.groupby('Class').mean())

#Undersampling
legit_sample = legit.sample(n=492)

#Concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())

print(new_dataset.groupby('Class').mean())

#Splitting the data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)
print(Y)

#Split the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model Training
model = LogisticRegression()
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

#Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)