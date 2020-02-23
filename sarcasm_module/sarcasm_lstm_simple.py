import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_lstm import AttentionMultiHead


class LSTMSarcasmSimple(nn.Module):

    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 output_size,
                 n_layers=2,
                 hidden_dim_lstm=256,
                 max_sentence_size=12,
                 dropout=0.1):
        super(LSTMSarcasmSimple, self).__init__()
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

        multi_head_attention = self.multi_head_attention(output)
        out = self.fc1(multi_head_attention)
        out_concat = out.transpose(1, 0).contiguous().view(input_vector.shape[0], -1)
        out = self.fc2(out_concat)
        out = self.dropout(out)
        out_sigmoid = self.sigmoid(out)
        return out_sigmoid
