import pandas as pd

messages = pd.read_csv(r"C:\Users\USER\Documents\AI\dataSets\SMSSpamCollection", sep= "\t", names= ["label", "message"])
print(messages)

# data cleaning and preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lm = WordNetLemmatizer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 5000)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 0)

# training model via naive bias model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()

spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('The accuracy for my spam classifier project is: ', acc)
