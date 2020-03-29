import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.naive_bayes import GaussianNB
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

data=pd.read_csv('food_reviews.csv')


def clean_tag(sentence):
    com1=re.compile('<.*?>')
    clean_tag = re.sub(com1,'',sentence)
    clean_tag = re.sub(r"won't", "will not", clean_tag)
    clean_tag = re.sub(r"can\'t", "can not", clean_tag)
    clean_tag = re.sub(r"n\'t", " not", clean_tag)
    clean_tag = re.sub(r"\'re", " are", clean_tag)
    clean_tag = re.sub(r"\'s", " is", clean_tag)
    clean_tag = re.sub(r"\'d", " would", clean_tag)
    clean_tag = re.sub(r"\'ll", " will", clean_tag)
    clean_tag = re.sub(r"\'t", " not", clean_tag)
    clean_tag = re.sub(r"\'ve", " have", clean_tag)
    clean_tag = re.sub(r"\'m", " am", clean_tag)
    return clean_tag

def clean_punch(sentence):
    wor1  = re.compile(r'[?|!|\'|"|#]')
    wor2 = re.compile(r'[.|,|)|(|\|/|;|&|-|:]')
    clean = re.sub(wor1,r'',sentence)
    clean = re.sub(wor2,r' ',clean)
    return clean
	
# declaring the snowball stemmer
sno = SnowballStemmer('english')

# creating the set of the stopwords
stop = set(stopwords.words('english'))

# cleaning the text 
final_sentence=[]
final_feat = []
for sente in tqdm(data['Text']):
    cleaned_words = clean_tag(sente)
    for words in clean_tag(cleaned_words).split():
        if (words.isalpha()) & (len(words) > 2):
            if(cleaned_words.lower() not in stop):
                s = sno.stem(cleaned_words.lower()).encode('utf8')
                final_feat.append(words.lower())
            else:
                continue
        else:
            continue
    final_sentence.append(final_feat)

# creating the Word2Vec model to convert the text into the numerical format	
model = gensim.models.Word2Vec(final_sentence,min_count = 5,size =50,workers =4)

# list the of vocabulary which is trained
word = list(model.wv.vocab)

sent_vectors= []

# using the Avg-W2V 
for sentence in tqdm(final_sentence):
    sent_vec = np.zeros(50)
    count = 0
    for words in sentence:
        if words in word:
            vec_word = model.wv[words]
            sent_vec += vec_word
            count += 1
    if count != 0:
        sent_vec /= count
        
    sent_vectors.append(sent_vec)

# Splitting the Dataset into train and test 
x_train,x_test,y_train,y_test = train_test_split(sent_vectors,data['Score'],test_size=0.3,random_state = 0)

# model creation
model = GaussianNB()
model.fit(x_train,y_train)
predi = model.predict(x_test)
print(confusion_matrix(y_true=y_test,y_pred=predi))
print(f1_score(y_true=y_test,y_pred=predi))