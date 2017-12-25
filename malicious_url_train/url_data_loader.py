import pandas as pd
import os


def load_url_data(data_dir_path):
    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',')

    url_data.columns = ['text', 'label']

    class_zero = url_data[url_data['label'] == 0].reset_index()
    class_one = url_data[url_data['label'] == 1].reset_index()

    class_zero = class_zero.truncate(before=1, after=class_one.shape[0])

    url_data = pd.concat([class_zero, class_one])
    url_data = url_data.sample(frac=1.0).reset_index()

    return url_data


def main():
    data_dir_path = './data'
    url_data = load_url_data(data_dir_path)

    print(url_data.head())


if __name__ == '__main__':
    main()
