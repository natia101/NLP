import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
import unicodedata

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')

df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")

df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0) #columna categorica la pasamos a 0 y 1

df['url'] = df['url'].str.lower() 

stop=stopwords.words('english')

def remove_stopwords(message):
    if message is not None:
        words = message.strip().split()
        words_filtered = []
        for word in words:
            if word not in stop:
                words_filtered.append(word)
        result = " ".join(words_filtered)  
    else:
        result=None
    return result

def clean_text_digits(texto):
    #'''Match all digits in the string and replace them by empty string.'''
    if texto is not None:
        pattern1 = r'[0-9]'
        pattern2 = '[^a-zA-Z]'
        pattern3 = "(\\d|\\W)+"
        pattern4 = r'http(s)'
        result = re.sub(pattern1, '', texto)
        result = re.sub(pattern2, '', result)
        result = re.sub(pattern3, '', result)
        result = re.sub(pattern4, '', result)
    else:
        result=None
    return result

#df['url']=df['url'].apply(remove_stopwords)
df['url']=df['url'].apply(clean_text_digits)

#limpio
def normalize_string(text_string):
    if text_string is not None:
        result = unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else:
        result = None
    return result

df['len_url'] = df['url'].apply(lambda x : len(x))
df['contains_subscribe'] = df['url'].apply(lambda x : 1 if "subscribe" in x else 0)
df['contains_hash'] = df['url'].apply(lambda x : 1 if "#" in x else 0)
df['num_digits'] = df['url'].apply(lambda x : len("".join(_ for _ in x if _.isdigit())) )
df['non_https'] = df['url'].apply(lambda x : 1 if "https" in x else 0)
df['num_words'] = df['url'].apply(lambda x : len(x.split("/")))

#target = 'is_spam'
#features = [f for f in df.columns if f not in ["url", target]]
#X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

message_vectorizer = CountVectorizer().fit_transform(df['url'])

X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['is_spam'], test_size = 0.45, random_state = 42, shuffle = True)

classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
print("SVM Accuracy Score -> ",accuracy_score(predictions, y_test)*100)

svm_accuracy_score = round(accuracy_score(predictions, y_test)*100,2)

print(f'Our model achieved {svm_accuracy_score}% accuracy!')