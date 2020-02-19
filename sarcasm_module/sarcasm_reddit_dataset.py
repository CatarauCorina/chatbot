import numpy as np
import pandas as pd
from torch.utils import data
import torch
import torch.nn as nn
import os, sys, re
#
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class RedditSarcasmDataset(data.Dataset):
    START_TOKEN = "<S>"
    END_TOKEN = "</S>"

    def __init__(self, **kwargs):
        self.stage = kwargs.get('stage', 'train')
        self.data_path = kwargs.get('path', '')
        self.max_seq_length = kwargs.get('max_seq', 10)
        self.stop_words = stopwords.words('english')
        self.freq_words = kwargs.get('frequent_word', [])
        # actual vocabulary
        self.voc = kwargs.get('voc_index', None)
        self.file_path = os.path.join(self.data_path, f'{self.stage}-balanced-sarcasm.csv')
        self.redit_csv = pd.read_csv(self.file_path)
        self.reddits = self.redit_csv['comment']
        self.labels = self.redit_csv['label']
        # vocabulary keys
        self.voc_keys = kwargs.get('voc_keys', [])
        self.start_token = self.START_TOKEN
        self.end_token = self.END_TOKEN
        self.idx = None
        self.filename = None
        return

    def __len__(self):
        return len(self.redit_csv['comment'][0])

    def tokenize_redit(self, s):
        s = word_tokenize(re.sub( '\W+', ' ', s.lower()))
        s = [w for w in s if w not in self.stop_words if w not in self.freq_words]
        ind = [self.voc[w] if w in self.voc_keys else self.voc['<UNK>'] for w in s]
        # pad or truncate the review
        if len(ind) >= self.max_seq_length:
            ind = ind[:self.max_seq_length]
        else:
            ind.extend([0]*(self.max_seq_length - len(ind)))
        # returns the list of indices
        ind.insert(0, self.voc[self.start_token])
        ind.append(self.voc[self.end_token])
        return ind

    def load_redit(self):
        for reddit in self.reddits:
            seq = torch.tensor(self.tokenize_redit(reddit), dtype=torch.int)
        return seq

    def __getitem__(self, idx):
        self.idx = idx
        # get data - tensor of integers
        self.filename = self.file_path
        if self.stage == 'test':
            self.ret_filename()
        seq = self.load_redit()
        # get label-split the file name on dot and subscript, take the second value
        split_review_name = re.split('[_.]', self.file_path)
        if int(split_review_name[-2]) > 6:
            lab = torch.tensor([1], dtype=torch.float)
        else:
            lab = torch.tensor([0], dtype=torch.float)
        return seq, lab

    def ret_filename(self):
        return self.filename

def main():
    data_args = {'path': '../../data_general/sarcasm/'}
    scr_data = RedditSarcasmDataset(**data_args)
    pars = {'shuffle': False, 'batch_size': 1}
    dataset_stage = data.DataLoader(scr_data, **pars)
    for el in dataset_stage:
        print(el.shape)
    return

if __name__ == '__main__':
    main()

