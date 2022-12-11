# -*- coding: utf-8 -*-

import os
import pandas as pd
from transformers import BertTokenizer
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
    """Extract text embeddings with Bert."""
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.load_data()

    def load_data(self):
        col_name = ['id', 'label', 'text']
        self.train = pd.DataFrame(columns=col_name)
        for file in os.listdir(self.dir_path):
            if file != 'twitter-2016test-A.tsv':
                one_df = pd.read_table(self.dir_path + file, 
                                       sep='\t',
                                       names=col_name,
                                       index_col=False)
                print(f'{file}: {one_df.shape[0]}')
                if file != 'test.tsv':
                    self.train = pd.concat([self.train, one_df])
                else:
                    self.test = one_df

        print(f'The number of train: {self.train.shape[0]}.')
        print(f'The number of test: {self.test.shape[0]}.')


if __name__ == '__main__':
#   clean_data()
#   model = BertModel(dir_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
