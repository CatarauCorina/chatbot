import os
import pickle
import pandas as pd
import numpy as np
from vocabulary import Vocabulary
import bz2
import unicodedata
import re


class DataCleaner:
    MAX_LENGTH_SENTENCE_PAIR = 10
    WORD_MIN_COUNT = 3

    def __init__(self, file_name):
        self.path = os.path.join("data", "")
        self.file_name = os.path.join(self.path, file_name)
        self.vocabulary = None
        self.pairs = None
        return

    @staticmethod
    def unicode_to_ascii(str_to_normalize):
        return ''.join(
            c for c in unicodedata.normalize('NFD', str_to_normalize)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(s):
        s = DataCleaner.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.,!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    @staticmethod
    def remove_twitter_symbols(csv, column_name):
        punctuations = '''!()-![]{};:+'"\,<>./?@#$%^&*_~'''
        csv[column_name] = csv[column_name].map(
            lambda x: x.strip(punctuations)
        )
        return csv

    @staticmethod
    def get_labels_and_texts(file, voc):
        labels = []
        texts = []
        for line in bz2.BZ2File(file):
            x = line.decode("utf-8")
            labels.append(int(x[9]) - 1)
            texts.append(x[10:].strip())
            voc.add_sentence(texts)
        return np.array(labels), texts


def main():
    data_path = "../../data_general/sarcasm/"
    voc = Vocabulary()
    ds_names = [('text_emotion.csv', 'sentiment'),
                ('train-balanced-sarcasm.csv', 'comment'), ('Sarcasm_Headlines_Dataset_v2.json', 'headline'),
                ('train.ft.txt.bz2', ''), ('test.ft.txt.bz2', '')]
    file_test = [('train.ft.txt.bz2', '')]
    for name, column_name in ds_names:
        print(f'Adding file {name}')
        file_name = os.path.join(data_path, name)
        sentences = []
        if name.endswith('.csv'):
            file = pd.read_csv(file_name)
            file = file[file[column_name].notna()]
            file = DataCleaner.remove_twitter_symbols(file, column_name)
            sentences = file[column_name]
        if name.endswith('.json'):
            file = pd.read_json(file_name, lines=True)
            file = file[file[column_name].notna()]
            file = DataCleaner.remove_twitter_symbols(file, column_name)
            sentences = file[column_name]
        if name.endswith('.bz2'):
            file_labels, file = DataCleaner.get_labels_and_texts(file_name, voc)
            sentences = []

        for sentence in sentences:
            voc.add_sentence(sentence)

    print('Trimming vocabulary')
    print(voc.num_words)
    voc.trim_vocabulary()
    print('After trim')
    print(voc.num_words)
    with open('voc_clean.pkl', 'wb') as f:
        pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

if __name__ == '__main__':
    main()


