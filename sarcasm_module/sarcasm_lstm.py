import torch.nn as nn
from attention_lstm import AttentionMultiHead


class LSTMSarcasm(nn.Module):

    def __init__(self,
                 vocabulary_size,
                 embedding_dim, hidden_size, output_size, n_layers=2, dropout=0.1):
        super(LSTMSarcasm, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        self.bi_lstm = nn.LSTM(
            input_size=embedding_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.multi_head_attention = AttentionMultiHead(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            nr_heads=8
        )
        self.fc1 = nn.Linear(embedding_dim, 100)
        self.fc2 = nn.Linear(100, output_size)
        self.dropout = nn.Dropout(dropout)
        self.log_sigmoid = nn.LogSigmoid()

        return

    def forward(self, input_vector):
        embedding = self.embedding(input_vector)
        lstm_result = self.bi_lstm(embedding)
        multi_head_attention = self.multi_head_attention(lstm_result)
        out = self.fc1(multi_head_attention)
        out = self.fc2(out)
        out = self.dropout(out)
        out_sigmoid = self.log_sigmoid(out)
        return out_sigmoid
