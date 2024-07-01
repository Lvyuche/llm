import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict
import re

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the model and metadata
model_info = torch.load('mlp_model.pth')
state_dict = model_info['state_dict']
seq_length = model_info['seq_length']
epochs = model_info['epochs']

# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# Create a simple dataset class to load vocabulary
class TextDataset:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        words = re.findall(r'\w+|[^\w\s]', text)
        self.word_to_idx = {"<UNK>": 0}  # Add UNK token
        self.word_to_idx.update({word: idx for idx, word in enumerate(set(words), 1)})
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

# Load dataset to get word_to_idx and idx_to_word
filepath = 'data/processed_articles.txt'
dataset = TextDataset(filepath)

print(f'Total vocabulary size: {dataset.vocab_size}')

# Create embedding matrix
embedding_dim = word2vec_model.vector_size
embedding_matrix = np.zeros((dataset.vocab_size, embedding_dim))
for word, idx in dataset.word_to_idx.items():
    if word in word2vec_model:
        embedding_matrix[idx] = word2vec_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
# Print the first three words and their embeddings
print("First ten words and their embeddings:")
for i, (word, idx) in enumerate(dataset.word_to_idx.items()):
    if i >= 3:
        break
    embedding = embedding_matrix[idx]
    print(f"Word: {word}, Embedding: {embedding[:3]}...")  # Print only the first 10 dimensions for readability


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
hidden_dim = 256  # Same as used in training

# Instantiate model and load state_dict
model = MLPModel(embedding_matrix, hidden_dim, seq_length).to(device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded from mlp_model.pth")

# Function to generate text using the loaded model
def generate_text(model, start_text, word_to_idx, idx_to_word, seq_length, n_words):
    input_seq = [word_to_idx.get(word, 0) for word in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_text[:]
    for _ in range(n_words):
        with torch.no_grad():
            output = model(input_seq)
        _, top_idx = torch.max(output, dim=1)
        predicted_word = idx_to_word.get(top_idx.item(), "<UNK>")
        generated_text.append(predicted_word)
        input_seq = torch.cat((input_seq[:, 1:], top_idx.unsqueeze(0)), dim=1)
    return ' '.join(generated_text)

# Test the model
start_text = "Machine learning is a field of study in artificial intelligence".split()
generated_text = generate_text(model, start_text, dataset.word_to_idx, dataset.idx_to_word, seq_length, 20)

print("Generated text:")
print(generated_text)
