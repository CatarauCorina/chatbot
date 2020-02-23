import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_lstm import AttentionMultiHead

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class LSTMSarcasm(nn.Module):

    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 output_size,
                 n_layers=2,
                 hidden_dim_lstm=64,
                 max_sentence_size=32,
                 dropout=0.1):
        super(LSTMSarcasm, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        self.fw_lstm_layers = []
        self.bw_lstm_layers = []
        input_dim_lstm = embedding_dim
        self.nr_layers = n_layers
        for layer in range(n_layers):
            lstm_lay_fw = nn.LSTM(
                input_size=input_dim_lstm,
                num_layers=1,
                hidden_size=hidden_dim_lstm,
                batch_first=True
             ).to(device)
            lstm_lay_bw = nn.LSTM(
                input_size=input_dim_lstm,
                num_layers=1,
                hidden_size=hidden_dim_lstm,
                batch_first=True
            ).to(device)
            input_dim_lstm = hidden_dim_lstm
            self.fw_lstm_layers.append(lstm_lay_fw)
            self.bw_lstm_layers.append(lstm_lay_bw)

        lstm_output_concat_size = 2*hidden_dim_lstm + 2*hidden_dim_lstm + embedding_dim
        self.multi_head_attention = AttentionMultiHead(
            input_size=lstm_output_concat_size,
            hidden_size=lstm_output_concat_size,
            nr_heads=8
        ).to(device)
        self.linear = nn.Linear(lstm_output_concat_size*2, hidden_dim_lstm)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim_lstm, 1)
        self.sigmoid = nn.Sigmoid()

        return

    def reverse_tensor(self, tensor_to_reverse):
        tensor_to_reverse = tensor_to_reverse.cpu().clone().detach()
        to_reve_numpy = np.flip(tensor_to_reverse.numpy(), 2).copy()  # Reverse of copy of numpy array of given tensor
        reversed_tensor = torch.from_numpy(to_reve_numpy)
        reversed_tensor = torch.tensor(reversed_tensor, device=device).requires_grad_(True)
        return reversed_tensor

    def forward(self, input_vector):
        embedding = self.embedding(input_vector)

        #output, (hidden, cell) = self.bi_lstm(embedding)
        lstm_input = embedding
        reversed_input = self.reverse_tensor(lstm_input)
        hidden_states = []
        for idx, layer in enumerate(self.fw_lstm_layers):
            fw = self.fw_lstm_layers[idx]
            bw = self.bw_lstm_layers[idx]
            output_fw, (hidden_fw, cell_fw) = fw(lstm_input)
            output_bw, (hidden_bw, cell_bw) = bw(reversed_input)
            concat_fw_bw = torch.cat([output_fw, output_bw], dim=2)
            hidden_states.append(concat_fw_bw)
            lstm_input = output_fw
            reversed_input = output_bw

        hidden_states.append(embedding)
        cat_fw_bw_em = torch.cat(hidden_states, dim=2)

        multi_head_attention = self.multi_head_attention(cat_fw_bw_em)
        avg_pool = torch.mean(multi_head_attention, 1)
        max_pool, _ = torch.max(multi_head_attention, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
