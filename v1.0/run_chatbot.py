#%%

import os
import torch
import torch.nn as nn
from chatbot import ChatBot

from data_preproc import DataPreprocessor, DataCleaner


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.device = device

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * 1
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(searcher, voc, sentence, chatbot, max_length=10):
    ### Format input sentence as a batch
    # words -> indexes
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    indexes_batch = [chatbot.data_loader.transform_sentence_word_indexes(sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc,chatbot):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = chatbot.dc.normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, chatbot)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def main():
    checkpoint_file = os.path.join('checkpoints', '6000_checkpoint.tar')
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    chatbot_dict = {
        'pretrained_embedding': True,
        'pretrained_embedding_file': embedding_sd,
        'pretrained_enc_dec': True,
        'pretrained_enc_file': encoder_sd,
        'pretrained_dec_file': decoder_sd
    }
    chatbot = ChatBot(**chatbot_dict)
    chatbot.dc.vocabulary.__dict__ = checkpoint['voc_dict']

    chatbot.encoder.eval()
    chatbot.decoder.eval()

    searcher = GreedySearchDecoder(chatbot.encoder, chatbot.decoder)

    evaluateInput(
        searcher=searcher,
        voc=chatbot.dc.vocabulary,
        chatbot=chatbot)
    return

if __name__ == '__main__':
    main()
