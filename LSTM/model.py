import torch.nn as nn
import torch
import numpy as np

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, embeddings):
        super(BiLSTMAttention, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        embedding_matrix = np.zeros((vocab_size, embed_size))
        for idx, vector in embeddings.items():
            embedding_matrix[idx] = vector
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 固定嵌入层权重

        # 双向LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 输出大小应与词汇表大小一致

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        
        # Bi-directional LSTM layer
        h_lstm, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(h_lstm), dim=1)
        context_vector = attn_weights * h_lstm
        context_vector = torch.sum(context_vector, dim=1)
        
        # Fully connected layer
        out = self.fc(context_vector)
        return out

if __name__ == "__main__":
    vocab_size = 10000  # 仅为示例，实际应为 len(dataset.vocab)
    embeddings = {i: np.random.rand(128) for i in range(vocab_size)}  # 伪造的嵌入向量
    model = BiLSTMAttention(vocab_size, embed_size=128, hidden_size=128, output_size=vocab_size, embeddings=embeddings)
    print(model)
