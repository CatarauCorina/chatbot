import torch
import spacy
import torch.nn as nn
import pickle
import itertools
import torch.functional as F
import torchtext
from torchtext.datasets import text_classification
from vocabulary import Vocabulary
from data_preproc import DataLoader, DataCleaner, DataPreprocessor

from torchtext.data import get_tokenizer

from sklearn.model_selection import train_test_split
MAX_FEATURES = 12000

tokenizer = get_tokenizer("basic_english")


def pad_input_batch(l):
    fill_value = 0
    return list(itertools.zip_longest(*l, fillvalue=fill_value))

def load_data():
    with open('train_.pkl', 'rb') as f:
        train_ = pickle.load(f)
    with open('test_.pkl', 'rb') as f:
        test_ = pickle.load(f)


    train_labels = train_[1]
    train_texts = train_[0]
    test_labels = test_[1]
    test_texts = test_[0]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, random_state=57643892, test_size=0.2)
    train_texts = [DataCleaner.normalize_string(s) for s in train_texts]
    voc = Vocabulary()
    [voc.add_sentence(review) for review in train_texts]
    dl = DataLoader(voc, train_texts)
    training_batches = dl.get_batches(
        size=128,
        nr_iterations=2, single=True)
    pad_batch = pad_input_batch(training_batches)

    # train_texts = pad_input_batch(torch.tensor([tokenizer(review) for review in train_texts]))
    #
    #
    # val_texts = pad_input_batch(torch.tensor([tokenizer(review) for review in val_texts]))
    # test_texts = pad_input_batch(torch.tensor([tokenizer(review) for review in test_texts]))
    # MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)
    # ds_train = text_classification.build_vocab_from_iterator(train_texts)

    return training_batches, train_labels, voc


class CNNSentiment(nn.Module):

    def __init__(self, input_size, embedding_size=64):
        super(CNNSentiment, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=(3, embedding_size))
        # self.batch_norm_1 = nn.BatchNorm1d()
        self.maxpool_1 = nn.MaxPool1d(3, stride=1)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(5, embedding_size))
        # self.batch_norm_2 = nn.BatchNorm1d()
        self.maxpool_2 = nn.MaxPool1d(5, stride=1)
        self.conv_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(5, embedding_size))
        self.global_max_pool = nn.MaxPool1d(5, embedding_size)


        return

    def forward(self, x):
        print(x.shape)
        out = self.embedding(x)
        print(out.shape)
        out = self.conv_1(out)
        print(out.shape)
        # out = self.batch_norm_1(out)
        print(out.shape)
        out = self.maxpool_1(out.squeeze())
        print(out.shape)
        out = self.conv_2(out)
        print(out.shape)
        # out = self.batch_norm_2(out)
        print(out.shape)
        out = self.maxpool_2(out)
        print(out.shape)
        out = self.conv_3(out)
        print(out.shape)
        out = self.global_max_pool(out)
        print(out.shape)
        return out

def main():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    batch_in, labels, voc = load_data()
    adam_params = {'lr': 0.001, 'weight_decay': 1e-4}
    print('params', adam_params)
    sentiment_model = CNNSentiment(voc.num_words)
    optimizer = torch.optim.Adam(sentiment_model.parameters(), **adam_params)

    for e in range(2):
        total_loss = 0
        num_batch = len(batch_in)
        print(num_batch)
        for id, batch in enumerate(batch_in):
            batch = batch.view(batch.shape[1], 1, batch.shape[0])
            # print (id, X.size() ,y.size())
            optimizer.zero_grad()
            X, y = batch.to(device), labels[id]
            # print(X, y)
            output = sentiment_model(X)
            # print(output.size(), y[:,0].size(), y)
            loss = F.binary_cross_entropy_with_logits(output, y.float())
            # loss = F.nll_loss(output, y[:,0].long())
            loss.backward()
            optimizer.step()
            total_loss += loss
            # print(id, loss)
    return


if __name__ == '__main__':
    main()

