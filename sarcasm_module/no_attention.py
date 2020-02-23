import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_lstm import AttentionMultiHead


class LSTMSarcasmNoAttention(nn.Module):

    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 output_size,
                 n_layers=2,
                 hidden_dim_lstm=64,
                 max_sentence_size=32,
                 dropout=0.1):
        super(LSTMSarcasmNoAttention, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        self.fw_lstm_layers = []
        self.bw_lstm_layers = []
        input_dim_lstm = embedding_dim
        self.nr_layers = n_layers
        self.bi_lstm = nn.LSTM(
            input_size=embedding_dim,
            num_layers=n_layers,
            hidden_size=hidden_dim_lstm,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim_lstm * 2*2, hidden_dim_lstm)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim_lstm, 1)
        self.sigmoid = nn.Sigmoid()

        lstm_output_concat_size = 2*hidden_dim_lstm
        self.multi_head_attention = AttentionMultiHead(
            input_size=lstm_output_concat_size,
            hidden_size=lstm_output_concat_size,
            nr_heads=8
        )
        self.fc1 = nn.Linear(lstm_output_concat_size, 100)
        self.fc2 = nn.Linear(100*max_sentence_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, input_vector):
        embedding = self.embedding(input_vector)
        embedding = embedding
        output, (hidden, cell) = self.bi_lstm(embedding)

        #multi_head_attention = self.multi_head_attention(output)
        avg_pool = torch.mean(output, 1)
        max_pool, _ = torch.max(output, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return self.sigmoid(out)
