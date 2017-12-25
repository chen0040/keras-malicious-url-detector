from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers.recurrent import LSTM
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

BATCH_SIZE = 64
EPOCHS = 10
VERBOSE = 1
NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256
EMBEDDING_SIZE = 100


def make_lstm_model(num_input_tokens, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=num_input_tokens, input_length=max_len, output_dim=EMBEDDING_SIZE))
    model.add(LSTM(NB_LSTM_CELLS, return_sequences=False, return_state=False, dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    np.random.seed(42)

    data_dir_path = './data'
    model_dir_path = './models'
    model_name = 'lstm-embed'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    architecture_file_path = model_dir_path + '/' + model_name + '-architecture.json'

    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',')
    url_data.columns = ['text', 'label']
    print(url_data.head())
    char2idx = dict()
    max_url_seq_length = 0
    for url in url_data['text']:
        max_url_seq_length = max(max_url_seq_length, len(url))
        for c in url:
            if c not in char2idx:
                char2idx[c] = len(char2idx)
    num_input_tokens = len(char2idx)
    idx2char = dict([(idx, c) for c, idx in char2idx.items()])

    config = dict()
    config['num_input_tokens'] = num_input_tokens
    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    config['max_url_seq_length'] = max_url_seq_length

    np.save(model_dir_path + '/' + model_name + '-config.npy', config)

    data_size = url_data.shape[0]
    X = np.zeros(shape=(data_size, max_url_seq_length))
    Y = np.zeros(shape=(data_size, 1))
    for i in range(data_size):
        url = url_data['text'][i]
        label = url_data['label'][i]
        for idx, c in enumerate(url):
            X[i, idx] = char2idx[c]
        Y[i] = label

    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = make_lstm_model(num_input_tokens, max_url_seq_length)
    open(architecture_file_path, 'w').write(model.to_json())

    checkpoint = ModelCheckpoint(weight_file_path)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train, test in kfold.split(X, Y):
        YtrainCount = Y[train].shape[0]
        YtestCount = Y[test].shape[0]
        Ytrain = np.zeros(shape=(YtrainCount, 2))
        Ytest = np.zeros(shape=(YtestCount, 2))
        for i in range(YtrainCount):
            label = int(Y[train][i][0])
            Ytrain[i, label] = 1
        for i in range(YtestCount):
            label = int(Y[test][i][0])
            Ytest[i, label] = 1

        model.fit(X[train], Ytrain, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
                  validation_data=(X[test], Ytest), callbacks=[checkpoint])

    model.save_weights(weight_file_path)


if __name__ == '__main__':
    main()
