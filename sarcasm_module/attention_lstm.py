import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AttentionLSTM(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.attention_vector = Parameter(torch.FloatTensor(hidden_size))
        return

    def forward(self, inputs, input_lengths):


