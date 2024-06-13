# preprocess.py
import nltk
import re
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())  # Convert to lowercase to ensure consistency
    return tokens

class QADataset(Dataset):
    def __init__(self, file_paths, max_len=512, embed_size=128, max_samples_per_file=1000):
        self.data = self.load_data(file_paths, max_samples_per_file)
        self.max_len = max_len
        self.embed_size = embed_size
        self.tokens = [preprocess_text(entry[1]) for entry in self.data]
        self.vocab, self.inv_vocab = self.build_vocab(self.tokens)  # Also store inverse vocab
        self.embeddings = self.train_word2vec(self.tokens)
        self.data = [(self.encode(entry[0]), self.encode(entry[1]), entry[2]) for entry in self.data]

    def load_data(self, file_paths, max_samples_per_file):
        data = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:max_samples_per_file]  # 只读取指定数量的行
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) == 3:  # 确保有三个部分
                    question, context, answer = parts
                    data.append((question, context, answer))
                else:
                    print(f"Skipping line: {line.strip()}")  # 打印被跳过的行，便于调试
        return data

    def build_vocab(self, tokens):
        vocab = {word: idx for idx, word in enumerate(set(sum(tokens, [])))}
        vocab['<UNK>'] = len(vocab)  # Add <UNK> token to vocabulary
        inv_vocab = {idx: word for word, idx in vocab.items()}
        return vocab, inv_vocab

    def train_word2vec(self, tokens):
        model = Word2Vec(sentences=tokens, vector_size=self.embed_size, window=5, min_count=1, workers=4)
        embeddings = {self.vocab[word]: model.wv[word] for word in model.wv.index_to_key}
        embeddings[self.vocab['<UNK>']] = np.zeros(self.embed_size)
        return embeddings

    def encode(self, text):
        tokens = preprocess_text(text)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids += [0] * (self.max_len - len(token_ids))  # Padding
        return torch.tensor(token_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, context, answer = self.data[idx]
        return question, context, answer

if __name__ == "__main__":
    # file_paths = ['data/qa_data.txt', 'data/squad_train.txt', 'data/squad_dev.txt']
    file_paths = ['data/qa_data.txt']
    max_samples_per_file = 500  # 设置每个文件读取的最大行数
    dataset = QADataset(file_paths, max_samples_per_file=max_samples_per_file)
    print("Vocabulary Size:", len(dataset.vocab))
    # print("Sample Data:", dataset[0])
    print("Embedding Dimension:", dataset.embed_size)
    
    # 可视化Word2Vec词嵌入
    words = list(dataset.vocab.keys())
    vectors = np.array([dataset.embeddings[dataset.vocab[word]] for word in words])
    print("Shape of the embedding matrix:", vectors.shape)
    
    # 使用t-SNE降维到3D
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], edgecolors='k', c='r')

    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word)

    ax.set_title("Word2Vec Embeddings Visualization using t-SNE (3D)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.show()
