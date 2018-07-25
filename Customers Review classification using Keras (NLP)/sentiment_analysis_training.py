import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, TimeDistributed
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')
EMBEDDING_FILE = '/home/balaji/Downloads/balaji/ds/word2vec/inputs/GoogleNews-vectors-negative300.bin'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    text = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]    
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return(text)

train=pd.read_csv("/home/balaji/Downloads/balaji/ds/word2vec/inputs/train.csv",delimiter='~')
X_train=train.Description
Y_train_data=train.Is_Response
texts_1=[]
labels=[]
for train_data in X_train:
    texts_1.append(text_to_wordlist(train_data))
for class_name in Y_train_data:
    if(class_name.lower()=='good'):
        labels.append(0)
    else:
        labels.append(1)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
word_index = tokenizer.word_index
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
nb_words = min(MAX_NB_WORDS, len(word_index))+1
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
model = Sequential()
model.add(Embedding(nb_words,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False))
model.add(LSTM(30,return_sequences=True,recurrent_dropout=0.2,unit_forget_bias=True))
model.add(Flatten())
model.add(Dense(30,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
print model.summary()
model.fit(data_1, labels, epochs=10,batch_size=32, validation_split=0.2,shuffle=True)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
