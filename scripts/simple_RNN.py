
from keras.models import Model
from keras.layers import Input,GRU,LSTM
import numpy as np

input_seq_len = 8
dim = 2
hidden_layers = 3

# consider this as one input sequence
X_input = np.random.randn(1, input_seq_len, dim)

# in this function we want to get all the hidden states of LSTM
def lstm1():
    input = Input(shape=(input_seq_len, dim))
    rnn = LSTM(hidden_layers, return_state=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h, c = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory
    # c: cell state : long term memory

    print("o:{}\nh:{}\nc:{}".format(o,h,c))


# in this function we want to get all the hidden states of LSTM
# and also return the sequences
def lstm2():
    input = Input(shape=(input_seq_len, dim))
    rnn = LSTM(hidden_layers, return_state=True, return_sequences=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h, c = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory
    # c: cell state : long term memory

    print("o:{}\nh:{}\nc:{}".format(o,h,c))



# in this function we want to get all the hidden states of GRU
def gru1():
    input = Input(shape=(input_seq_len, dim))
    rnn = GRU(hidden_layers, return_state=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    # remember GRU returns only 2 values. output and hidden state
    o, h = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory

    print("o:{}\nh:{}".format(o,h))


# in this function we want to get all the hidden states and sequences of GRU
def gru2():
    input = Input(shape=(input_seq_len, dim))
    rnn = GRU(hidden_layers, return_state=True, return_sequences=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h = model.predict(X_input)
    # o: output of lstm
    # h: hidden state of lstm: short term memory

    print("o:{}\nh:{}".format(o,h))

print("-----LSTM 1------")
lstm1()
print("-----LSTM 2------")
lstm2()
print("-----GRU 1------")
gru1()
print("-----GRU 2------")
gru2()