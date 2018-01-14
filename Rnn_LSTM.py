import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


# 1. Loading the data
print("loading data...")

pos_file_name = "pos_amazon_cell_phone_reviews.json"
neg_file_name = "neg_amazon_cell_phone_reviews.json"
pos_file = open(pos_file_name, "r")
neg_file = open(neg_file_name, "r")
pos_data = json.loads(pos_file.read())['root']
neg_data = json.loads(neg_file.read())['root']
print("Posititve data loaded. ", len(pos_data), "entries")
print("Negative data loaded. ", len(neg_data), "entries")

print("done loading data...")

plabels = []
nlabels = []

# 2.Process reviews into sentences
pos_sentences, neg_sentences = [], [] 
for entry in pos_data :
    pos_sentences.append(entry['summary'] + " . " + entry['text'])
    plabels.append(1)
for entry in neg_data :
    nlabels.append(0)
    neg_sentences.append(entry['summary'] + " . " + entry['text'])
print(len(pos_sentences))
print(len(neg_sentences))

texts = pos_sentences + neg_sentences 
labels = [1]*len(pos_sentences) + [0]*len(neg_sentences)

# print("after app", labels)

#print(type(pos_sentences), pos_sentences.shape, type(neg_sentences), neg_sentences.shape)
#print(type(texts), texts.shape, type(labels), labels.shape) 

# 3. Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 50

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# print(labels)

labels = np.array(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

rest_data = data[3000:]
rest_labels = labels[3000:]


data = data[:5000]
labels = labels[:5000]

VALIDATION_SPLIT = 0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

print(data.shape, labels.shape, nb_validation_samples)

print(labels)

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print(len(x_train), len(y_train))

#GloVe
embeddings_index = {}
f = open('glove.6B/glove.6B.50d.txt', 'r', encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = MAX_SEQUENCE_LENGTH

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 


# Training the LSTM model

batch_size = 128

model = Sequential()

model.add(embedding_layer)

model.add(LSTM(64))

model.add(Dropout(0.50))

model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs

model.compile('adam', 'binary_crossentropy', metrics=['accuracy', precision, recall])

print('Train...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=16,
          validation_data=[x_val, y_val])

x = model.evaluate(rest_data[:5000], rest_labels[:5000])
print("Loss: ", x[0])
print("Accuracy: ", x[1])
print("Precision: ", x[2])
print("Recall: ", x[3])
