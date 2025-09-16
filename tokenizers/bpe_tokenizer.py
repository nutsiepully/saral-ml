"""
Minimal Byte Pair Encoding Tokenizer implementation, based on GPT-2
"""

class BPETokenizer:

    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def _get_pair_counts(self, ids):
        pair_counts = {}
        for pair in zip(ids[0:-1], ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def _merge_tokens(self, ids, merge_pair, merge_id):
        new_ids = []

        i = 0
        while i < len(ids):
            if (i + 1) < len(ids) and (ids[i], ids[i + 1]) == merge_pair:
                new_ids.append(merge_id)
                i += 2
                continue

            new_ids.append(ids[i])
            i+=1

        return new_ids

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = text.encode("utf-8")

        vocab = { id: bytes([id]) for id in range(256) }
        merges = {}
        max_id = 255
        
        for i in range(num_merges):
            # Unlikely case there's only 1 token left after merges.
            if len(ids) <= 1:
                break

            pair_counts = self._get_pair_counts(ids)
            max_pair = max(pair_counts, key=pair_counts.get)

            max_id += 1
            vocab[max_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            merges[max_pair] = max_id
            ids = self._merge_tokens(ids, max_pair, max_id)

        self.vocab = vocab
        self.merges = merges


    def encode(self, text):
        """
        Takes an input text, and returns an encoded set of IDs based on learnt vocabulary.
        """
        ids = text.encode("utf-8")

        while True:
            pair_counts = self._get_pair_counts(ids)
            print(pair_counts)
            min_pair = min(pair_counts, key = lambda k: self.merges.get(k, float("inf")))
            print(min_pair)

            if not self.merges.get(min_pair):
                # No pair exists in vocabulary, merging is done.
                break

            print(ids)
            ids = self._merge_tokens(ids, min_pair, self.merges[min_pair])
            print(ids)
        
        return ids


    def decode(self, ids):
        """
        Decodes a set of token IDs into text based on learnt vocabulary.
        """
        text_bytes = b"".join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8")

    def __str__(self):
        return "Vocab: {} \n\nMerges: {}\n".format(self.vocab, self.merges)
