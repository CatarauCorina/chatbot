import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import ast
import unicodedata
import codecs
from io import open
from vocabulary import Vocabulary
import itertools
import math


class DataPreprocessor:
    FILES = {
        'char_meta': 'movie_characters_metadata.txt',
        'movie_conv': 'movie_conversations.txt',
        'movie_lines': 'movie_lines.txt',
        'movie_titles_meta': 'movie_titles_metadata.txt',
        'movie_raw': 'raw_script_urls.txt'
    }
    MOVIE_LINES_FIELDS = ["line_id", "character_id", "movie_id", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1_id", "character2_id", "movie_id", "utterance_ids"]

    def __init__(self, data_path="", movie_line_fields=None, movie_conversation_fields=None):
        self.corpus = os.path.join("data", data_path)
        self.lines = {}
        self.separator = '+++$+++'
        self.conversations = []
        if movie_line_fields is None:
            self.movie_line_fields = self.MOVIE_LINES_FIELDS
        if movie_conversation_fields is None:
            self.movie_conversation_fields = self.MOVIE_CONVERSATIONS_FIELDS
        self.process_dialog_lines(self.FILES['movie_lines'])
        self.group_dialog_lines_into_conversation(self.FILES['movie_conv'])
        return

    def process_dialog_lines(self, file_name):
        lines = self.lines
        with open(os.path.join(self.corpus, file_name), 'r', encoding='iso-8859-1') as f:
            for line in f:
                split_text = line.split(self.separator)
                line_obj = {}
                for i, field in enumerate(self.movie_line_fields):
                    line_obj[field] = split_text[i].strip()
                lines[line_obj['line_id']] = line_obj
        self.lines = lines
        return lines

    def group_dialog_lines_into_conversation(self, file_name):
        with open(os.path.join(self.corpus, file_name), 'r', encoding='iso-8859-1') as f:
            for line_file in f:
                split_text = line_file.split(self.separator)
                conv_dict = {}
                for i, field in enumerate(self.movie_conversation_fields):
                    value = split_text[i]
                    conv_dict[field] = value
                    if field == 'utterance_ids':
                        value = ast.literal_eval(split_text[i].strip())
                        conv_dict['lines_ids'] = value
                        conv_dict['lines'] = [self.lines[id_line] for id_line in conv_dict['lines_ids']]
                self.conversations.append(conv_dict)
        return self.conversations

    def create_response_reply(self):
        qa_pairs = []
        for conversation in self.conversations:
            for i in range(len(conversation["lines"]) - 1):
                input_line = conversation["lines"][i]["text"].strip()
                response_line = conversation["lines"][i+1]["text"].strip()
                if input_line and response_line:
                    qa_pairs.append([input_line, response_line])
        return qa_pairs

    def write_to_file(self):
        print("\nWriting newly formatted file...")
        delimiter = '\t'
        file_name = 'formatted_movie_lines.txt'
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))
        with open(os.path.join(self.corpus, file_name), 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            qa_sentences = self.create_response_reply()
            for pair in qa_sentences:
                writer.writerow(pair)
        return file_name

    def printLines(self, file, n=10):
        with open(file, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)
        return


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
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def read_vocabulary(self, corpus_name=""):
        print("Reading lines...")

        lines = open(self.file_name, encoding='utf-8'). \
            read().strip().split('\n')

        pairs = [[DataCleaner.normalize_string(s) for s in l.split('\t')] for l in lines]
        voc = Vocabulary(corpus_name)
        return voc, pairs

    def filter_short_qas(self, pair):
        return len(pair[0].split(' ')) < self.MAX_LENGTH_SENTENCE_PAIR and len(pair[1].split(' ')) < self.MAX_LENGTH_SENTENCE_PAIR

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_short_qas(pair)]

    def clean_data_pipeline(self, filter_pairs=True):
        vocabulary, pairs = self.read_vocabulary()
        print(f"Read {len(pairs)} sentence pairs")
        if filter_pairs:
            pairs = self.filter_pairs(pairs)
        print(f"Trimmed long sentences {len(pairs)} sentence pairs")
        for pair in pairs:
            vocabulary.add_sentence(pair[0])
            vocabulary.add_sentence(pair[1])
        print(f"Counted words: {vocabulary.num_words}")
        self.vocabulary = vocabulary
        self.pairs = pairs
        return self

    def trim_rare_words(self):
        self.vocabulary = self.vocabulary.trim_vocabulary(self.WORD_MIN_COUNT)
        keep_pairs_without_trimmed_words = []
        for pair in self.pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            for word in input_sentence.split(' '):
                if word not in self.vocabulary.word2index:
                    keep_input = False
                    break

            for word in output_sentence.split(' '):
                if word not in self.vocabulary.word2index:
                    keep_output = False
                    break

            if keep_input and keep_output:
                keep_pairs_without_trimmed_words.append(pair)
        print(f"Trimmed from {len(self.pairs)} pairs to {len(keep_pairs_without_trimmed_words)} of total")
        self.pairs = keep_pairs_without_trimmed_words
        return self


class DataLoader:

    def __init__(self, vocabulary, pairs):
        self.vocabulary = vocabulary
        self.pairs = pairs
        return

    def transform_sentence_word_indexes(self, sentence):
        return [self.vocabulary.word2index[word] for word in sentence.split(' ')] \
               + [self.vocabulary.EOS_token]

    def pad_input_batch(self, l):
        fill_value = self.vocabulary.PAD_token
        return list(itertools.zip_longest(*l, fillvalue=fill_value))

    def define_input_batch(self, sentences):
        indexes_batch = [self.transform_sentence_word_indexes(sentence) for sentence in sentences]
        unpaded_lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = self.pad_input_batch(indexes_batch)
        return torch.LongTensor(pad_list), unpaded_lengths

    def binary_mask_output(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.vocabulary.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def define_output_batch(self, sentences):
        indexes_batch = [self.transform_sentence_word_indexes(sentence) for sentence in sentences]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = self.pad_input_batch(indexes_batch)
        mask = self.binary_mask_output(pad_list)
        mask = torch.BoolTensor(mask)
        pad_var = torch.LongTensor(pad_list)
        return pad_var, mask, max_target_len

    def get_batch_single(self, batch):
        inp, lengths = self.define_input_batch(batch)
        return inp

    def get_batch(self, pairs_batch):
        pairs_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pairs_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.define_input_batch(input_batch)
        output, mask, max_target_len = self.define_output_batch(output_batch)
        return inp, lengths, output, mask, max_target_len

    def get_batches(self, size=5, nr_iterations=10, single=False):
        if single:
            list_batches = [
                self.get_batch_single([random.choice(self.pairs) for _ in range(size)])
                for _ in range(nr_iterations)]
        else:
            list_batches = [
                self.get_batch([random.choice(self.pairs) for _ in range(size)])
                for _ in range(nr_iterations)]
        return list_batches


def main():
    dp = DataPreprocessor()
    file_name_formatted = dp.write_to_file()
    dc = DataCleaner(file_name_formatted)
    dc.clean_data_pipeline().trim_rare_words()
    data_loader = DataLoader(dc.vocabulary, dc.pairs)
    data_loader.get_batches()
    return


if __name__ == '__main__':
    main()
