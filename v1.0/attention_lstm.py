import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AttentionMultiHead(nn.Module):

    def __init__(self, input_size, hidden_size, nr_heads):
        super(AttentionMultiHead, self).__init__()
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttention(input_size, hidden_size) for idx_head in range(nr_heads)])
        self.linear_out = nn.Linear(nr_heads*hidden_size, input_size)
        return

    def forward(self, input_vector):
        all_heads = []
        for head in self.heads:
            out = head(input_vector)
            all_heads.append(out)
        z_out_concat = torch.cat(all_heads, dim=2)
        z_out_out = F.relu(self.linear_out(z_out_concat))
        return z_out_out


class SelfAttention(nn.Module):

    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.key_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.value_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.softmax = nn.Softmax()
        return

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))

        value_out = F.relu(self.value_linear(input_vector))
        out_q_k = torch.div(torch.bmm(query_out, key_out.transpose(1, 2)), math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine


def main():
    in_vect = torch.rand(2, 4)
    at_multi = AttentionMultiHead(4, 3, 8)
    at_multi(in_vect)
    return

if __name__ == '__main__':
    main()