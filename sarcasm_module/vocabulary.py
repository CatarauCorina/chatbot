class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name=""):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.PAD_token: "PAD",
            self.SOS_token: "SOS",
            self.EOS_token: "EOS"
        }
        self.num_words = 3
        return

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
        return

    def init_vocabulary(self, words):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            self.PAD_token: "PAD",
            self.SOS_token: "SOS",
            self.EOS_token: "EOS"
        }
        self.num_words = 3  # Count default tokens

        for word in words:
            self.add_word(word)
        return

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
        return

    def trim_vocabulary(self, min_count=10):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        self.init_vocabulary(keep_words)
        return self
