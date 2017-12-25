from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    np.random.seed(42)
    
    data_dir_path = './data'
    model_dir_path = './models'
    model_name = 'lstm'
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

    np.save(model_dir_path + '/' + model_name + '-config.npy', config)

    data_size = url_data.shape[0]
    X = np.zeros(shape=(data_size, max_url_seq_length, num_input_tokens))
    Y = np.zeros(shape=(data_size, 2))
    for i in range(data_size):
        url = url_data['text'][i]
        label = url_data['label'][i]
        for idx, c in enumerate(url):
            X[i, idx, char2idx[c]] = 1
        Y[i, label] = 1

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)


if __name__ == '__main__':
    main()
