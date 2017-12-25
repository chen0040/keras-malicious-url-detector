import numpy as np
import pandas as pd
import os
from malicious_url_train.lstm_train import make_lstm_model


class LstmPredictor(object):

    model = None
    num_input_tokens = None
    idx2char = None
    char2idx = None
    max_url_seq_length = None

    def __init__(self):
        pass

    def load_model(self, config_file_path, weight_file_path):
        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        self.model = make_lstm_model(self.num_input_tokens)
        self.model.load_weights(weight_file_path)

    def predict(self, url):
        data_size = 1
        X = np.zeros(shape=(data_size, self.max_url_seq_length, self.num_input_tokens))
        for idx, c in enumerate(url):
            if c in self.char2idx:
                X[0, idx, self.char2idx[idx]] = 1
        predicted = self.model.predict(X)[0]
        predicted_label = np.argmax(predicted)
        return predicted_label


def main():
    model_name = 'lstm'
    data_dir_path = '../malicious_url_train/data'
    model_dir_path = '../malicious_url_train/models'
    config_file_path = model_dir_path + '/' + model_name + '-config.npy'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    predictor = LstmPredictor()
    predictor.load_model(config_file_path=config_file_path, weight_file_path=weight_file_path)

    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',')
    url_data.columns = ['text', 'label']
    for url in url_data['text']:
        predicted_label = predictor.predict(url)
        actual_label = url_data['label']
        print('predicted: ' + predicted_label + ' actual: ' + actual_label)


if __name__ == '__main__':
    main()
