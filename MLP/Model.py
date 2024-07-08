import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
import re

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create a simple dataset
class TextDataset(Dataset):
    def __init__(self, filepath, seq_length):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        words = re.findall(r'\w+|[^\w\s]', text)
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

# Prepare data
filepath = 'data/processed_articles.txt'
seq_length = 10
dataset = TextDataset(filepath, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print total vocabulary size
print(f'Total vocabulary size: {dataset.vocab_size}')

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

# Define MLP model with more layers and neurons
class MLPModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, seq_length):
        super(MLPModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.fc1 = nn.Linear(embedding_dim * seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # Convert word indices to embedding vectors
        x = x.view(x.size(0), -1)  # Flatten the embedding vectors
        x = self.fc1(x)  # Fully connected layer 1
        x = self.relu(x)  # ReLU activation function
        x = self.fc2(x)  # Fully connected layer 2
        x = self.relu(x)  # ReLU activation function
        x = self.fc3(x)  # Fully connected layer 3
        x = self.relu(x)  # ReLU activation function
        x = self.fc4(x)  # Fully connected layer 4
        x = self.relu(x)  # ReLU activation function
        x = self.fc5(x)  # Fully connected layer 5, output predictions
        return x

# Model parameters
hidden_dim = 256  # Increased number of neurons

# Instantiate model, loss function, and optimizer
model = MLPModel(embedding_matrix, hidden_dim, seq_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 30
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

# Save the model
model_info = {
    'state_dict': model.state_dict(),
    'seq_length': seq_length,
    'epochs': epochs,
    'word_to_idx': dataset.word_to_idx,
    'idx_to_word': dataset.idx_to_word
}

# Print the first three words and their embeddings
print("First three words and their embeddings:")
for i, (word, idx) in enumerate(dataset.word_to_idx.items()):
    if i >= 3:
        break
    embedding = embedding_matrix[idx]
    print(f"Word: {word}, Embedding: {embedding[:3]}...")  # Print only the first 3 dimensions for readability

torch.save(model_info, 'mlp_model_' + str(hidden_dim) + 'dim_ ' + str(epoch + 1) + 'epoch.pth')
print("Model saved to mlp_model.pth")
