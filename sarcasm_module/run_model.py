import torch
from torch.utils import data

import torch.nn.functional as F
from sarcasm_reddit_dataset import RedditSarcasmDataset
from sarcasm_lstm import LSTMSarcasm
from sarcasm_lstm_simple import LSTMSarcasmSimple
from no_attention import LSTMSarcasmNoAttention

from torch.utils.tensorboard import SummaryWriter


def train_model():
    run_server = False
    writer = SummaryWriter('sarcasm/model_b32_lr001')
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    if run_server:
        data_args = {'path': './'}
    else:
        data_args = {'path': '../../data_general/sarcasm/'}
    scr_data = RedditSarcasmDataset(**data_args)
    pars = {'shuffle': False, 'batch_size': 31}
    dataset_stage = data.DataLoader(scr_data, **pars)

    adam_params = {'lr': 0.0001, 'weight_decay': 1e-4}
    model = LSTMSarcasmNoAttention(scr_data.voc.num_words, embedding_dim=100, output_size=1)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **adam_params)

    for e in range(3):
        total_loss = 0
        count_batches = 0
        for idx, batch in enumerate(dataset_stage):
            x, y = batch
            if pars['batch_size'] == 1:
                y = torch.unsqueeze(y, 0)
            else:
                y = torch.unsqueeze(y, dim=1)
            optimizer.zero_grad()
            x = torch.tensor(x, device=device)
            y = torch.tensor(y, device=device)

            output = model(x)
            loss = F.binary_cross_entropy(output, y.float())
            loss.backward()
            print(loss.item())
            writer.add_scalar(f'Loss batches epoch {e}', loss.item(), idx)

            optimizer.step()
            total_loss += loss
            count_batches += 1
        loss_epoch = total_loss/count_batches
        writer.add_scalar('Loss epoch', loss_epoch, e)
    return


def main():
    train_model()
    return


if __name__ == '__main__':
    main()
