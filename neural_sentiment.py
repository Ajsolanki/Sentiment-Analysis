
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Bidirectional
from keras.layers import LSTM
import numpy as np
import pandas as pd

seed = 7
np.random.seed(seed)


train,test = pd.read_csv('cleantrain.csv'),pd.read_csv('test.csv')
def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        #X_batch = X_data[index_batch,:].toarray()
        X_batch = X_data[index_batch,:]
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            np.random.shuffle(index)
            counter=0
            
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=100000))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(train['text'], train['sentiment'], test_size=0.30)

# tvec = TfidfVectorizer(stop_words=None,max_features=100000, ngram_range=(1, 3))
tvec = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1,3),
                        stop_words=None,
                        lowercase=False,
                        max_features=100000,
                        norm='l2',
                        sublinear_tf=True)
x_train_tfidf = tvec.fit_transform(x_train).toarray()
x_validation_tfidf = tvec.fit_transform(x_validation)

from sklearn.preprocessing import Normalizer
norm = Normalizer().fit(x_train_tfidf)
x_train_tfidf_norm = norm.transform(x_train_tfidf)
x_validation_tfidf_norm = norm.transform(x_validation_tfidf)

model.fit_generator(generator=batch_generator(x_train_tfidf_norm, y_train, 32),
                    epochs=15, validation_data=(x_validation_tfidf_norm, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/32)