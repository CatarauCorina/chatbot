import numpy as np
import pandas as pd
from torch.utils import data
import torch
import torch.nn as nn
import os, sys, re
#
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vocabulary import Vocabulary


class RedditSarcasmDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.stage = kwargs.get('stage', 'train')
        self.data_path = kwargs.get('path', '')
        self.max_seq_length = kwargs.get('max_seq', 30)
        self.stop_words = stopwords.words('english')
        self.freq_words = kwargs.get('frequent_word', [])
        # actual vocabulary
        self.file_path = os.path.join(self.data_path, f'{self.stage}-balanced-sarcasm.csv')
        self.redit_csv = pd.read_csv(self.file_path)
        res_not_na = self.redit_csv[self.redit_csv['comment'].notna()]
        res_lt_100 = self.redit_csv[self.redit_csv['comment'].str.len() < 100]

        self.reddits = res_lt_100['comment']

        self.voc = Vocabulary(f'{self.stage}-balanced-sarcasm.csv')

        self.labels = res_lt_100['label']
        self.create_reddit_vocabulary()
        # vocabulary keys
        self.voc_keys = self.voc.word2index.keys()
        self.start_token = self.voc.SOS_token
        self.end_token = self.voc.EOS_token
        self.idx = None
        self.filename = None
        return

    def __len__(self):
        return len(self.reddits)

    def tokenize_redit(self, s):
        s = word_tokenize(re.sub( '\W+', ' ', s.lower()))
        s = [w for w in s if w not in self.stop_words if w not in self.freq_words]
        ind = [self.voc.word2index[w] for w in s if w in self.voc_keys]
        # pad or truncate the review
        if len(ind) >= self.max_seq_length:
            ind = ind[:self.max_seq_length]
        else:
            ind.extend([0]*(self.max_seq_length - len(ind)))
        # returns the list of indices
        ind.insert(0, self.start_token)
        ind.append(self.end_token)
        return ind

    def create_reddit_vocabulary(self):
        for reddit in self.reddits:
            self.voc.add_sentence(reddit)
        return

    def load_redit(self, idx):
        tokenized_reddit = torch.tensor(self.tokenize_redit(self.reddits.iloc[idx]), dtype=torch.long)
        return tokenized_reddit

    def __getitem__(self, idx):
        self.idx = idx
        self.filename = self.file_path
        if self.stage == 'test':
            self.ret_filename()
        seq = self.load_redit(idx)
        return seq, self.labels.iloc[idx]

    def ret_filename(self):
        return self.filename


def main():
    data_args = {'path': '../../data_general/sarcasm/'}
    scr_data = RedditSarcasmDataset(**data_args)
    pars = {'shuffle': False, 'batch_size': 1}
    dataset_stage = data.DataLoader(scr_data, **pars)
    for el in dataset_stage:
        x, y = el
        print(x.shape)
    return

if __name__ == '__main__':
    main()

