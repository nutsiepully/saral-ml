from bpe_tokenizer import BPETokenizer

t = BPETokenizer()
t.train("hello, world. lo and behlod, the wor is ld", 260)
tokens = t.encode("hello")

print(t)
print("Encoded Tokens: ", tokens)
print("Decoded tokens: ", t.decode(tokens))
