import pandas as pd
import os


def main():
    data_dir_path = './data'
    url_data = pd.read_csv(data_dir_path + os.path.sep + 'URL.txt', sep=',')

    url_data.columns = ['text', 'label']
    print(url_data.head())


if __name__ == '__main__':
    main()
