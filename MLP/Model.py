import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np

# Create a simple dataset
class TextDataset(Dataset):
    def __init__(self, filepath, seq_length):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        words = text.split()
        self.word_to_idx = {"<UNK>": 0}  # Add UNK token
        self.word_to_idx.update({word: idx for idx, word in enumerate(set(words), 1)})
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        self.seq_length = seq_length
        self.data = self.preprocess(words)

    def preprocess(self, words):
        data = []
        for i in range(len(words) - self.seq_length):
            seq_in = words[i:i + self.seq_length]
            seq_out = words[i + self.seq_length]
            data.append((seq_in, seq_out))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_in, seq_out = self.data[idx]
        x = torch.tensor([self.word_to_idx.get(word, 0) for word in seq_in], dtype=torch.long)
        y = torch.tensor(self.word_to_idx.get(seq_out, 0), dtype=torch.long)
        return x, y

# Define MLP model
class MLPModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, seq_length):
        super(MLPModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.fc1 = nn.Linear(embedding_dim * seq_length, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert word indices to embedding vectors
        x = x.view(x.size(0), -1)  # Flatten the embedding vectors
        x = self.fc1(x)  # Fully connected layer 1
        x = self.relu(x)  # ReLU activation function
        x = self.fc2(x)  # Fully connected layer 2, output predictions
        return x

# Prepare data
filepath = 'data/helloworld.txt'
seq_length = 5
dataset = TextDataset(filepath, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# Create embedding matrix
embedding_dim = word2vec_model.vector_size
embedding_matrix = np.zeros((dataset.vocab_size, embedding_dim))
for word, idx in dataset.word_to_idx.items():
    if word in word2vec_model:
        embedding_matrix[idx] = word2vec_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Model parameters
hidden_dim = 128

# Instantiate model, loss function, and optimizer
model = MLPModel(embedding_matrix, hidden_dim, seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

# Test the model
def predict(model, words, word_to_idx, idx_to_word, seq_length, n_words):
    model.eval()
    input_seq = [word_to_idx.get(word, 0) for word in words]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
    predicted_text = words[:]
    for _ in range(n_words):
        with torch.no_grad():
            output = model(input_seq)
        _, top_idx = torch.max(output, dim=1)
        predicted_word = idx_to_word[top_idx.item()]
        predicted_text.append(predicted_word)
        input_seq = torch.cat((input_seq[:, 1:], top_idx.unsqueeze(0)), dim=1)
    return ' '.join(predicted_text)

start_text = "hello world and student is"
predicted_text = predict(model, start_text.split(), dataset.word_to_idx, dataset.idx_to_word, seq_length, 20)
print(predicted_text)
