import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from math import log, sqrt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

train = pd.read_csv('training.csv', encoding = 'latin-1')
#print(train.head(3))
test = pd.read_csv('metaSample.csv', encoding = 'latin-1')
#print(test.head(2))

#Print Excel shape (num of rows and coloumns):
print(train.shape)
print(test.shape)

#Check for duplicates: test.drop_duplicates(inplace = True)

#Show missing data values
#print(train.isnull().sum())
#print(test.isnull().sum())
test = test.dropna(subset=['Comment'])
test = test.drop('Username', axis='columns')
test = test.drop('num', axis='columns')
test = test.drop('Platform', axis='columns')
print(test.shape)
#print(test.isnull().sum())

#Process Text
def process_text(text):
	#Remove Punctuation
	nopunc = [char for char in text if char not in string.punctuation]
	nopunc = ''.join(nopunc)

	#Remove Stopwords (Useless words or data)
	clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
	return clean

#print(train.head(1))
#print(test.head(1))
train['text'].head().apply(process_text)
test['Comment'].head().apply(process_text)

#text = " ".join(i for i in train.text)
#stopword = stopwords.words('english')
#wordcloud = WordCloud(stopwords=stopword, background_color="white").generate(text)
#plt.figure( figsize=(15,10))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

#Convert Text to matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
vectorize = CountVectorizer(analyzer=process_text)
messages_bow = vectorize.fit_transform(train['text'])

#Split data for 80% training 20% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(messages_bow, train['label'], test_size = 0.10, random_state = 0)

#Get the shape of messages_bow
#print(messages_bow.shape)

#Create and train the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_train, y_train)

#Print the predictions
#print(classifier.predict(x_train))
#Print actual values
#print(y_train.values)

#Evaluate the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(x_train)
print(classification_report(y_train, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: ', accuracy_score(y_train, pred))


#Print the predictions
#print(classifier.predict(x_test))
#Print actual values
#print(y_test.values)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(x_test)
print(classification_report(y_test, pred))
print()
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
print()
print('Accuracy: ', accuracy_score(y_test, pred))

games_bow = vectorize.transform(test['Comment'])
pred2 = np.round(np.clip(classifier.predict_proba(games_bow), 0, 1))
test['label'] = pred2
test.to_csv('output.csv', mode='a', header=False)

#test(pred2, columns=['label']).to_csv('output.csv', mode='a', header=False)
#print(pred2)
#pred2 = encoder.inverse_transform(pred)


