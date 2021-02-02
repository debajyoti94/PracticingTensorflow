from keras.models import Model
from keras.layers import Input,Bidirectional,LSTM
import numpy as np

input_seq_len = 8
dim = 2
hidden_layers = 3

# consider this as one input sequence
X_input = np.random.randn(1, input_seq_len, dim)

# in this function we want to get all the hidden states of LSTM
def bi_lstm1():
    # functional model, not sequential
    input = Input(shape=(input_seq_len, dim))
    rnn = Bidirectional(LSTM(hidden_layers, return_state=True, return_sequences=True))
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h1, c1, h2,c2 = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory
    # c: cell state : long term memory

    print("o:{}\nh1:{}\nc1:{}".format(o,h1,c1))
    print("h2:{}\nc2:{}".format( h2, c2))


# in this function we want to get all the hidden states of LSTM
# and also return the sequences
def bi_lstm2():
    input = Input(shape=(input_seq_len, dim))
    rnn = Bidirectional(LSTM(hidden_layers, return_state=True, return_sequences=False))
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h1, c1, h2, c2 = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory
    # c: cell state : long term memory

    print("o:{}\nh1:{}\nc1:{}".format(o, h1, c1))
    print("h2:{}\nc2:{}".format(h2, c2))

bi_lstm1()