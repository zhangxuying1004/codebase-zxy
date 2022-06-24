
class HyperParameter:
    def __init__(self):
        # the most frequent words
        self.total_words = 10000
        self.max_sentence_len = 80

        self.embedding_len = 100

        self.epochs = 30
        self.batch_sz = 128
        self.learning_rate = 4e-3