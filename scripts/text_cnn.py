
# here we will do some text preprocessing for toxic comments challenge

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer # this will be used during preprocessing
from keras.preprocessing.sequence import pad_sequences # in order to create the fix length matrix

# for modeling
from keras.layers import Dense,MaxPool1D,Embedding
from keras.layers import GlobalMaxPool1D,Input,Conv1D
from keras.models import Model

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


#  here we read the data
raw_toxic_df = pd.read_csv('../input/toxic_comments/train.csv', delimiter=',',
                          encoding='UTF-8')

comments = raw_toxic_df['comment_text'].values
possible_labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
targets = raw_toxic_df[possible_labels].values
print(targets)


# here we start converting the strings to integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True,
                      split=' ')
# now we assign an index value to each word in the dataset
tokenizer.fit_on_texts(comments)
# replace the words by indices in the sequences
sequences = tokenizer.texts_to_sequences(comments)


# print(sequences)
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

# prepare the embedding matrix
num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
embedding_matrix = np.zeros(shape=(num_words, EMBEDDING_DIM))

for word, index in word2idx.items():
    if index < num_words:
        embedding_vector = glove_mapping.get(word)
        # using .get() as it handles exception if the word doesn't exist in the dictionary
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

print(embedding_matrix.shape)

# here we create the model
# couple of things as recap:
# data = num comments x max sequence length
# targets = num comments x 6 :6-> possible labels
# embedding matrix = vocab size x embedding dimension

# making the embedding layer
embedding_layer = Embedding(num_words, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False) # as we do not want these weights to be updated during training

# creating the model here
input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,)) # shape of one sequence
x = embedding_layer(input_layer)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPool1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPool1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPool1D(3)(x)
x = GlobalMaxPool1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_layer, output)
model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

# training model here
model.fit(data, targets, batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT)

model.save_weights('../models/text_cnn.h5')

plt.title("Train vs Validation loss")
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.legend()
plt.grid()
plt.savefig('../plots/train_valid_loss_textcnn.png')