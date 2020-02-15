import torch.nn as nn


class LSTMSarcasm(nn.Module):

    def __init__(self, vocabulary_size, embedding_dim, hidden_size, output_size, n_layers=2, dropout=0.1):
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

        return

    def forward(self, input_vector):
        embedding = self.embedding(input_vector)
        lstm_result = self.bi_lstm(embedding)
        return
