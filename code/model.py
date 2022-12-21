# -*- coding: utf-8 -*-

import os
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf


dir_path = '../../semeval-2017-tweets_Subtask-A/downloaded/'


def clean_data():
    one_file = dir_path + 'twitter-2016test-A.tsv'
    test_file = dir_path + 'test.tsv'
    test_lst = list()
    with open(one_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            line = line[:3]
            test_lst.append('\t'.join(line) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lst)


class BertModel():
    """Extract text embeddings with Bert.

    Args:
        dir_path: Path of datasets.
        seed: Random seed.
    """
    def __init__(self, dir_path, seed=42):
        self.dir_path = dir_path
        self.seed = seed
        self.load_data()

    def load_data(self):
        col_name = ['id', 'label', 'text']
        data = pd.DataFrame(columns=col_name)

        for file in os.listdir(self.dir_path):
            if file != 'twitter-2016test-A.tsv':
                one_df = pd.read_table(self.dir_path + file, 
                                       sep='\t',
                                       names=col_name,
                                       index_col=False)
                print(f'{file}: {one_df.shape[0]}')
                data = pd.concat([data, one_df])

        self.train, self.test = train_test_split(data, test_size=0.2, random_state=self.seed)
        print(f'The number of train: {self.train.shape[0]}.')
        print(f'The number of test: {self.test.shape[0]}.')


if __name__ == '__main__':
#   clean_data()
    model = BertModel(dir_path)
