import os
import torch
import random
import torch.nn as nn
from torch import optim
from encoder import EncoderRNN
from decoder import DecoderRNN
from attention import AttentionRNN
from data_preproc import DataPreprocessor, DataCleaner, DataLoader
from torch.utils.tensorboard import SummaryWriter


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class ChatBot:

    def __init__(self, **kwargs):
        dp = DataPreprocessor()
        file_name_formatted = dp.write_to_file()
        dc = DataCleaner(file_name_formatted)
        dc.clean_data_pipeline().trim_rare_words()
        self.data_loader = DataLoader(dc.vocabulary, dc.pairs)
        self.dp = dp
        self.dc = dc
        load_embedding = kwargs.get('pretrained_embedding', False)
        embedding_file = kwargs.get('pretrained_embedding_file', None)

        load_enc_dec = kwargs.get('pretrained_enc_dec', False)
        load_enc_file = kwargs.get('pretrained_enc_file', None)
        load_dec_file = kwargs.get('pretrained_dec_file', None)

        self.model_name = kwargs.get('model_name', 'cb_model')
        attn_model = kwargs.get('attention_type', 'dot')
        self.hidden_size = kwargs.get('hidden_size', 500)
        self.encoder_nr_layers = kwargs.get('enc_nr_layers', 2)
        self.decoder_nr_layers = kwargs.get('dec_nr_layers', 2)
        dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 64)
        self.clip = kwargs.get('clip', 50.0)
        self.teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 1.0)
        self.learning_rate = kwargs.get('lr', 0.0001)
        self.decoder_learning_ratio = kwargs.get('decoder_learning_ratio', 5.0)
        self.nr_iteration = kwargs.get('nr_iterations', 4000)
        self.print_every = kwargs.get('print_every', 1)
        self.save_every = 500
        self.embedding = nn.Embedding(self.dc.vocabulary.num_words, self.hidden_size)
        if load_embedding:
            self.embedding.load_state_dict(embedding_file)
        # Initialize encoder & decoder models
        encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_nr_layers, dropout)
        decoder = DecoderRNN(
            attn_model,
            self.embedding,
            self.hidden_size,
            self.dc.vocabulary.num_words,
            self.decoder_nr_layers,
            dropout
        )

        if load_enc_dec:
            encoder.load_state_dict(load_enc_file)
            decoder.load_state_dict(load_dec_file)
        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
        return

    def mask_nllloss(self, inp, target, mask):
        n_total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, n_total.item()

    def train_pair(self, **kwargs):

        max_target_len = kwargs.get('max_target_len')

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = kwargs.get('input')
        target_variable = kwargs.get('output')
        lengths = kwargs.get('input_lengths')
        mask = kwargs.get('output_mask')

        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.dc.vocabulary.SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < kwargs.get('teacher_forcing_ratio', 0.2) else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_nllloss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_nllloss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train(self, **kwargs):
        writer = SummaryWriter(f'chatbot/trimmed_{self.model_name}')
        training_batches = self.data_loader.get_batches(
            size=self.batch_size,
            nr_iterations=self.nr_iteration)

        print('Initializing ...')
        start_iteration = 1
        print_loss = 0

        print("Training...")
        for iteration in range(start_iteration, self.nr_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            input_train_pair = {
                'input': input_variable,
                'input_lengths': lengths,
                'output': target_variable,
                'output_mask': mask,
                'max_target_len': max_target_len,

            }
            loss = self.train_pair(**input_train_pair)
            print_loss += loss

            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / self.nr_iteration * 100, print_loss_avg))
                writer.add_scalar('Loss ', print_loss_avg, iteration)
                print_loss = 0

            # Save checkpoint
            if (iteration % self.save_every == 0):
                directory = os.path.join('models', self.model_name, 'cornell', '{}-{}_{}'.format(
                    self.encoder_nr_layers,
                    self.decoder_nr_layers, self.hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.dc.vocabulary.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def main():
    chatbot = ChatBot()
    chatbot.train()
    return

if __name__ == '__main__':
    main()