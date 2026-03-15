import numpy as np
import random
from collections import Counter

class DataProcessor:
    def __init__(self, filepath, vocab_size=10000, window_size=2, num_neg_samples=5):
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        
        # mapping ids for future eval
        self.word2idx = {}
        self.idx2word = {}
        # distribution for the subsampling
        self.word_counts = {}
        # pos examples
        self.corpus_ids = []
        # neg examples
        self.neg_sample_table = []

    def prepare_data(self, num_chars=1000000):
        with open(self.filepath, 'r') as f:
            text = f.read(num_chars)
        words = text.split()
        words = words[:-1]
        
        counts = Counter(words)
        top_words = counts.most_common(self.vocab_size - 1)
        
        # unk token for new words edge case
        self.word2idx['<UNK>'] = 0
        self.idx2word[0] = '<UNK>'
        self.word_counts[0] = 0
        
        for idx, (word, count) in enumerate(top_words, start=1):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[idx] = count
            
        raw_ids = [self.word2idx.get(w, 0) for w in words]
        
        self.corpus_ids = self._subsample(raw_ids)
        self._build_neg_sample_table()

    def _subsample(self, raw_ids):
        total_words = len(raw_ids)
        t = 1e-5
        
        drop_probs = {}
        for idx, count in self.word_counts.items():
            freq = count / total_words
            drop_probs[idx] = max(0, 1 - np.sqrt(t / freq)) if freq > 0 else 0  # P(drop) = 1 - sqrt(t / freq)
            
        return [w_id for w_id in raw_ids if random.random() > drop_probs.get(w_id, 0)]

    def _build_neg_sample_table(self, table_size=1000000):
        pow_freqs = np.array(list(self.word_counts.values())) ** 0.75
        probs = pow_freqs / np.sum(pow_freqs)
        
        self.neg_sample_table = np.random.choice(
            list(self.word_counts.keys()), 
            size=table_size, 
            p=probs
        )

    def generate_batches(self, batch_size):
        batch_centers = []
        batch_contexts = []
        
        for i in range(len(self.corpus_ids)):
            center_word = self.corpus_ids[i]
            
            start = max(0, i - self.window_size)
            end = min(len(self.corpus_ids), i + self.window_size + 1)
            
            context_words = self.corpus_ids[start:i] + self.corpus_ids[i+1:end]
            
            for context_word in context_words:
                batch_centers.append(center_word)
                batch_contexts.append(context_word)
                
                if len(batch_centers) == batch_size:
                    negatives = np.random.choice(
                        self.neg_sample_table, 
                        size=(batch_size, self.num_neg_samples)
                    )
                    
                    yield (
                        np.array(batch_centers, dtype=np.int32), 
                        np.array(batch_contexts, dtype=np.int32), 
                        negatives
                    )
                    batch_centers = []
                    batch_contexts = []
