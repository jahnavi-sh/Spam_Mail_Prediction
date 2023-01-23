#First, understanding the problem statement - 
#Spam mail is unwanted junk mail which is sent our in bulk to an indiscriminant recipient list. It is mainly used for commercial 
#purposes. Spam emails sent to penetrate user inboxes with messages intended to promote products and services in order to turn a 
#profit. A report from estimates some 281.1 billion emails are sent every day, worldwide. That’s 37 emails for every person on 
#the planet. And out of all that mails, more than half is spam. It wastes our time, impacts productivity. It is also a severe messaging
#threat as it opens us up to phishing and malware attacks.

#Aim of this project is to build a machine learning model that correctly predicts spam model from a dataset of model. So that 
#we can separate the junk mail from the important mails. 

#Workflow for the project - 
#1. load the mail data 
#2. data preprocessing 
#3. train test split 
#4. model used - logistic regression model 
#5. model evaluation - accuracy score 

#importing the libraries 
#linear algebra - for building matrices 
import numpy as np 

#data preprocessing and exploration 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

#model building and evaluation 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the data 
#from a csv file to a pandas dataframe 
raw_mail_data = pd.read_csv(r'mail_data.csv')

#the dataset contains 5572 rows (5573 data points) and 2 columns 

#the columns in the dataset are as follows - 
#1. Category - ham or spam 
#2. Message - This containes the textual content of the mail 

#replace the null value with a null string 
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

#view the first five rows of the dataframe 
mail_data.head()

#check the total number of rows and columns 
mail_data.shape

#label encoding 
#label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

#separating the data as texts and label 
X = mail_data['Message']
Y = mail_data['Category']

#splitting the data into test data and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)

#transform the text data into feature vectors that can be used as input to the logistic regression model 
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.fit_transform(X_test)

#convert Y_train and Y_test values as integers 
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#training the model 
model = LogisticRegression()

#fit the model in training data 
model.fit(X_train_features, Y_train)

#evaluate the results of model on training data 
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print ('acccuracy on training data = ', accuracy_on_training_data)
#accuracy score of the model on training data is 0.967

#fit model on test data 
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print ('acccuracy on test data = ', accuracy_on_test_data)
#accuracy score of the model on test data is 0.965

#the accuracy score for training data and test data are very close to each other. Therefore, the model is performing very well. 