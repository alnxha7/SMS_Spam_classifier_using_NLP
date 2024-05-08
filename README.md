# SMS_Spam_classifier_using_NLP

## Introduction
A Natural Language Processing with SMS Data to predict whether the SMS is Spam/Ham with multinomial-naive-bayes and using various data cleaning and processing techniques like PorterStemmer,CountVectorizer. I got an maximum accuracy of 98% from this project

## Used dataset
I used the SMS_SpamCollection dataset to train my model, which can be accessed via the link below:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Reviewing the results of the outputs of our trained model
The accuracy of our Naïve Bayes multinomial model is 98.206278 % The Precision of our Naïve Bayes multinomial model is 93.2098765 % The Recall of our Naïve Bayes multinomial model is 94.375 %

## confusion matrix of this model

![202918749-a4700297-d395-4b0e-99ca-060a270b4e69](https://github.com/alnxha7/SMS_Spam_classifier_using_NLP/assets/129566733/697f4f93-76e5-4b39-903a-2ef8f188caea)

## Steps
Import libraries
Upload dataset
Create the data frame
Split the data
Vectorize the data
Train & predict
Calculate accuracy, precision, and recall
calculate the confusion matrix
Test the model with a new Sms/Email massage
