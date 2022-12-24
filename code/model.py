# -*- coding: utf-8 -*-

import os
import random
import evaluate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import tensorflow as tf
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from datasets import load_dataset


dir_path = '../../semeval-2017-tweets_Subtask-A/downloaded/'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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
        tune: Hyperparameter tuning mode. Default: False.
        batch: Batch size. Default: 16.
        epoch: The number of epoch. Default: 50.
        seed: Random seed. Default: 42.
    """
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
    names = ['bert-base-cased', 'roberta-base', 'distilbert-base-cased']
    
    def __init__(self, dir_path, tune=False, batch=16, epoch=50, seed=42):
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
            train_dev, val_dev = train_test_split(self.train, test_size=0.2, random_state=self.seed)
            train_dev.to_csv(dir_path + 'hug-train-dev.csv', index=False)
            val_dev.to_csv(dir_path + 'hug-val-dev.csv', index=False)
        
        if tune:
            self.data = load_dataset('csv', data_files={'train': dir_path + 'hug-train-dev.csv', 'test': dir_path + 'hug-val-dev.csv'})
        else:
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

    def _report(self, preds, trues, info):
        string = f"""
        {info}
        Acc: {accuracy_score(trues, preds)}
        Recall: {recall_score(trues, preds, average='macro')}
        F1: {f1_score(trues, preds, average='macro')}
        Report:{classification_report(trues, preds)}
        """
        print(string)
        with open('../results/report.txt', 'a') as f:
            f.write(string)

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    def bert(self, lr=1e-5, bert_type='bert-base-cased', info='', prob=False):
        """The template for Bert models."""
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        def preprocess_function(examples):
            return tokenizer(examples['text'], truncation=True)
        tokenized_data = self.data.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')

        batches_per_epoch = len(tokenized_data['train']) // self.batch
        total_train_steps = int(batches_per_epoch * self.epoch)
        optimizer, schedule = create_optimizer(init_lr=lr, num_warmup_steps=0, num_train_steps=total_train_steps)
        model = TFAutoModelForSequenceClassification.from_pretrained(
                bert_type, num_labels=3, id2label=BertModel.id2label, label2id=BertModel.label2id
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
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f'../models/{bert_type}', save_weights_only=True, monitor='accuracy', save_best_only=True
                )
        callbacks = [metric_callback, early_stopping_callback, model_checkpoint_callback]
        history = model.fit(x=tf_train_set, epochs=self.epoch, callbacks=callbacks)

        model.load_weights(f'../models/{bert_type}')
        preds = model.predict(tf_test_set)
        if prob:
            np.save(f'../results/{bert_type + info}-prob.npy', preds.logits)
        preds = np.argmax(preds.logits, axis=1)
        np.save(f'../results/{bert_type + info}.npy', preds)
        return history

    def tuning(self, bert_type='bert-base-cased'):
        """Hyperparameter tuning."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2]
        for lr in lrs:
            history = self.bert(lr=lr, bert_type=bert_type, info=f'_{lr}')
            his_df = pd.DataFrame.from_dict(history.history)
            his_df.to_csv(f'../results/{bert_type}_{lr}.csv', index=False)
            
    def proposed_hard_voting(self):
        """Proposed model: Hard voting."""
        def vote(a, b, c):
            vote_dict = {0: 0, 1: 0, 2: 0}
            lst = [a, b, c]
            for element in lst:
                for key in list(vote_dict.keys()):
                    if element == key:
                        vote_dict[key] += 1
            for item in list(vote_dict.items()):
                if item[1] >= 2:
                    return item[0]
            return random.sample(list(vote_dict.keys()), 1)[0]

        preds_dict = dict()
        for name in BertModel.names:
            preds = np.load(f'../results/{name}.npy')
            preds_dict[name] = preds.tolist()
        preds_df = pd.DataFrame.from_dict(preds_dict)
        preds = preds_df.apply(lambda row : vote(row['bert-base-cased'], row['roberta-base'], row['distilbert-base-cased']), axis=1).to_numpy()
        trues = np.array(self.data['test']['label'])
        self._report(preds, trues, 'Proposed Hard Voting')

    def proposed_soft_voting(self):
        """Proposed model: Soft voting."""
        preds_lst = list()
        for name in BertModel.names:
            preds = np.load(f'../results/{name}-prob.npy')
            preds_lst.append(normalize(preds))
        preds = np.average(np.array(preds_lst), axis=0)
        preds = np.argmax(preds, axis=1)
        trues = np.array(self.data['test']['label'])
        self._report(preds, trues, 'Proposed Soft Voting')

    def main(self, bert_type='bert-base-cased'):
        """Run Bert models."""
        history = self.bert(bert_type=bert_type, prob=True)
        his_df = pd.DataFrame.from_dict(history.history)
        his_df.to_csv(f'../results/{bert_type}.csv', index=False)

    def evaluation(self):
        """Model evaluation."""
        trues = np.array(self.data['test']['label'])
        for name in BertModel.names:
            preds = np.load(f'../results/{name}.npy')
            self._report(preds, trues, name)


if __name__ == '__main__':
    clean_data()

    # Hyperparameter tuning
    model = BertModel(dir_path, tune=True)
    model.tuning()
    model.tuning(bert_type='roberta-base')
    model.tuning(bert_type='distilbert-base-cased')

    # Prediction and evaluation
    model = BertModel(dir_path)
    model.main()
    model.main(bert_type='roberta-base')
    model.main(bert_type='distilbert-base-cased')
    model.evaluation()
    model.proposed_hard_voting()
    model.proposed_soft_voting()
