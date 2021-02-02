# solve the same toxic comments using lstm

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding,Dense,LSTM,Input, GlobalMaxPool1D
from keras.models import Model

# load the pretrained embeddings
# for configuration
MAX_SEQUENCE_LENGTH = 100     # based on observatoions
MAX_VOCAB_SIZE = 20000        # based on scientific experiments
EMBEDDING_DIM = 300           # since we are using pre-trained embeddings
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

# loading word vectors
glove_mapping = {}
with open('../glove_embeddings/model.txt') as f:

    for line in f:
        values = line.split()
        # print(values[0])
        try:
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_mapping[word] = vector

        except ValueError:
            print("Skipping word {}".format(word))
            continue

    print("Total words vectors found={}".format(len(glove_mapping)))


# load the dataset
raw_toxic_df = pd.read_csv('../input/toxic_comments/train.csv', delimiter=',',
                          encoding='UTF-8')

comments = raw_toxic_df['comment_text'].values
possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
targets = raw_toxic_df[possible_labels].values
print(targets)


# fit the tokenizer to texts
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True,
                      split=' ')
# now we assign an index value to each word in the dataset
tokenizer.fit_on_texts(comments)
# replace the words by indices in the sequences
sequences = tokenizer.texts_to_sequences(comments)


# fit the tokenizer to sequences
# figuring out the length of sequences
print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print("median sequence length:", s[len(s) // 2])


# get word to index mapping
word2idx = tokenizer.word_index

# pad sequences to create a fixed size data matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("shape of data tensor={}".format(data.shape))



# create the embedding matrix
num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
embedding_matrix = np.zeros(shape=(num_words, EMBEDDING_DIM))

for word, index in word2idx.items():
    if index < num_words:
        embedding_vector = glove_mapping.get(word)
        # using .get() as it handles exception if the word doesn't exist in the dictionary
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

print(embedding_matrix.shape)

# create the embedding layer
embedding_layer = Embedding(num_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False) # as we do not want these weights to be updated during training

# create the functional model
input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_layer)
x = LSTM(15, return_sequences=True, activation='relu')(x) # we want to get the hidden states for every point in time
x = GlobalMaxPool1D()(x)             # this is to consider each feature in time and keep rack of the most important ones
output = Dense(len(possible_labels), activation='sigmoid')(x) # activation is sigmoid bcoz of multilabel classification

model = Model(input_layer, output)

# train the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, targets, batch_size=BATCH_SIZE,
          epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

# get the loss
model.save_weights('../models/text_cnn.h5')

plt.title("Train vs Validation loss")
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.legend()
plt.grid()
plt.savefig('../plots/train_valid_loss_lstm.png')