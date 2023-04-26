# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:05:51 2023

@author: dhair
"""
#*************** Import Libraries ****************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

#************* Import DataSet ***************

df_emails = pd.read_csv('/Users/dhair/OneDrive/Desktop/emails.csv')
print(df_emails)
print(df_emails.describe())
print(df_emails.info())

#*********** Finding the Ham or Spam msgs ****************

ham = df_emails[df_emails['spam'] == 0 ]
print(ham.info() , '/n')

spam = df_emails[df_emails['spam'] == 1 ]
print(spam.info(),'/n')

#************** Calculate the Percentage **********

print('Spam Percentage : ' ,  (len(spam)/len(df_emails) )*100 , '%' )

print('Ham Percentage : ' , (len(ham)/len(df_emails) )*100 , '%'  )

sns.countplot(df_emails['spam'] , label = "count spam vs ham ")
plt.show()

#*************** Apply the CountVectorizer ************

vectorizer = CountVectorizer()
email_vectorizer = vectorizer.fit_transform(df_emails['text'])
print(email_vectorizer.shape)

print(vectorizer.get_feature_names())

print(email_vectorizer.toarray())

label = df_emails['spam'].values

print('Labels : ' , label)

#************ Apply the Naive Bayes **************

NB_classifier = MultinomialNB()
NB_classifier.fit(email_vectorizer , label )

#*********** Creating the X and Y datasets ************

X = email_vectorizer
Y = label

#*********** Train Test Split ***************

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.25 , random_state = 0 )
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train , Y_train)

#************ Calculate the confusion matrix and classification report *********
#************ Apply on Training set ************

y_pred = NB_classifier.predict(X_train)
cm = confusion_matrix(Y_train, y_pred)
print ('Confusion Matrix : ' , cm)
sns.heatmap(cm, annot = True )
plt.show()

#************* Apply on Testing Set ************

Y_pred_test = NB_classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred_test)
print ('Confusion Matrix : ' , cm)
sns.heatmap (cm  , annot = True)
plt.show()

#************* Print the Classification Report ***************

print('Classification Report For Train Data : ' , classification_report(Y_train, y_pred))

print('Classification Report for Test Data : ' , classification_report(Y_test, Y_pred_test))

























