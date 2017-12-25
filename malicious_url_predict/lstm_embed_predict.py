import numpy as np

from malicious_url_train.lstm_embed_train import make_lstm_embed_model
from malicious_url_train.url_data_loader import load_url_data


class LstmEmbedPredictor(object):
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

        self.model = make_lstm_embed_model(self.num_input_tokens, self.max_url_seq_length)
        self.model.load_weights(weight_file_path)

    def predict(self, url):
        data_size = 1
        X = np.zeros(shape=(data_size, self.max_url_seq_length))
        for idx, c in enumerate(url):
            if c in self.char2idx:
                X[0, idx] = self.char2idx[c]
        predicted = self.model.predict(X)[0]
        predicted_label = np.argmax(predicted)
        return predicted_label


def main():
    model_name = 'lstm-embed'
    data_dir_path = '../malicious_url_train/data'
    model_dir_path = '../malicious_url_train/models'
    config_file_path = model_dir_path + '/' + model_name + '-config.npy'
    weight_file_path = model_dir_path + '/' + model_name + '-weights.h5'
    predictor = LstmEmbedPredictor()
    predictor.load_model(config_file_path=config_file_path, weight_file_path=weight_file_path)

    url_data = load_url_data(data_dir_path)
    count = 0
    for url, label in zip(url_data['text'], url_data['label']):
        predicted_label = predictor.predict(url)
        print('predicted: ' + str(predicted_label) + ' actual: ' + str(label))
        count += 1
        if count > 20:
            break


if __name__ == '__main__':
    main()
