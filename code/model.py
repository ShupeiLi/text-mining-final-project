# -*- coding: utf-8 -*-

import os
import evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from datasets import load_dataset


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
        batch: Batch size. Default: 16.
        epoch: The number of epoch. Default: 5.
        seed: Random seed. Default: 42.
    """
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def __init__(self, dir_path, batch=16, epoch=5, seed=42):
        self.dir_path = dir_path
        self.batch = batch
        self.epoch = epoch
        self.seed = seed
        if not os.path.exists(dir_path + 'hug-train.csv'):
            self._load_data()
            self._preprocessing()
            self.train.drop(columns='id', inplace=True)
            self.test.drop(columns='id', inplace=True)
            self.train.to_csv(dir_path + 'hug-train.csv', index=False)
            self.test.to_csv(dir_path + 'hug-test.csv', index=False)
        self.data = load_dataset('csv', data_files={'train': dir_path + 'hug-train.csv', 'test': dir_path + 'hug-test.csv'})
        self.accuracy = evaluate.load('accuracy')

    def _load_data(self):
        """Load datasets and split training / test sets."""
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

    def _preprocessing(self):
        """Preprocess text."""
        def text_pre(col):
            col = col.str.replace(r'(@.*?)[\s]', ' ', regex=True)
            col = col.str.replace(r'&amp', '&', regex=True)
            col = col.str.replace(r'\s+', ' ', regex=True)
            return col.str.strip()
        
        self.train['text'] = text_pre(self.train['text'])
        self.test['text'] = text_pre(self.test['text'])
        self.train['label'] = self.train['label'].map(BertModel.label2id)
        self.test['label'] = self.test['label'].map(BertModel.label2id)

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    def base_bert(self):
        """Bert base cased model."""
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        def preprocess_function(examples):
            return tokenizer(examples['text'], truncation=True)
        tokenized_data = self.data.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')

        batches_per_epoch = len(tokenized_data['train']) // self.batch
        total_train_steps = int(batches_per_epoch * self.epoch)
        optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
        model = TFAutoModelForSequenceClassification.from_pretrained(
                'bert-base-cased', num_labels=3, id2label=BertModel.id2label, label2id=BertModel.label2id
                )
        tf_train_set = model.prepare_tf_dataset(
                tokenized_data['train'],
                shuffle=True,
                batch_size=self.batch,
                collate_fn=data_collator,
                )
        tf_test_set = model.prepare_tf_dataset(
                tokenized_data['test'],
                shuffle=False,
                batch_size=self.batch,
                collate_fn=data_collator,
                )
        model.compile(optimizer=optimizer)
        metric_callback = KerasMetricCallback(metric_fn=self._compute_metrics, eval_dataset=tf_test_set)
        model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=metric_callback)


if __name__ == '__main__':
#   clean_data()
    model = BertModel(dir_path)
    model.base_bert()
