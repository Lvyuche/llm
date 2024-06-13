import torch

def save_vocab(vocab, path):
    with open(path, 'w') as f:
        for word, idx in vocab.items():
            f.write(f'{word}\t{idx}\n')

def load_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    return vocab
