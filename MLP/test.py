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
word_to_idx = model_info['word_to_idx']
idx_to_word = model_info['idx_to_word']

# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# Create embedding matrix
embedding_dim = word2vec_model.vector_size
embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
for word, idx in word_to_idx.items():
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
hidden_dim = 256  # Same as used in training

# Instantiate model and load state_dict
model = MLPModel(embedding_matrix, hidden_dim, seq_length).to(device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded from mlp_model.pth")

# Function to build N-Gram model
def build_ngram_model(text, n):
    ngrams = defaultdict(int)
    words = re.findall(r'\w+|[^\w\s]', text)
    for i in range(len(words) - n):
        ngram = tuple(words[i:i + n])
        ngrams[ngram] += 1
    return ngrams

# Define the filepath variable
filepath = 'data/processed_articles.txt'

# Build N-Gram model from text
with open(filepath, 'r', encoding='utf-8') as file:
    text = file.read()
ngram_model = build_ngram_model(text, seq_length)

# Implementing Beam Search with n-gram language model
def beam_search_with_ngram(model, start_text, word_to_idx, idx_to_word, seq_length, n_words, beam_width=3, ngram_model=None, ngram_weight=0.1):
    model.eval()
    input_seq = [word_to_idx.get(word, 0) for word in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    sequences = [[list(input_seq[0].cpu().numpy()), 0.0]]  # Initialize sequences with the start text and log-probability 0

    for _ in range(n_words):
        all_candidates = []
        for seq, score in sequences:
            seq_tensor = torch.tensor(seq[-seq_length:], dtype=torch.long).unsqueeze(0).to(device)
            print(f"Input shape: {seq_tensor.shape}")
            with torch.no_grad():
                output = model(seq_tensor)
            log_probs, indices = torch.topk(torch.nn.functional.log_softmax(output, dim=1), beam_width)
            for i in range(beam_width):
                candidate = [seq + [indices[0][i].item()], score - log_probs[0][i].item()]

                # Apply n-gram language model adjustment
                if ngram_model is not None and len(candidate[0]) >= seq_length:
                    ngram = tuple(candidate[0][-seq_length:])
                    ngram_score = ngram_model.get(ngram, 0)
                    candidate[1] += ngram_weight * ngram_score

                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    # Return top sequence
    top_sequence = sequences[0][0]
    top_predicted_text = [idx_to_word.get(idx, "<UNK>") for idx in top_sequence]
    return ' '.join(top_predicted_text)

# Test the model
start_text = "Machine learning is a field of study in artificial intelligence".split()
generated_text = beam_search_with_ngram(model, start_text, word_to_idx, idx_to_word, seq_length, 20, beam_width=3, ngram_model=ngram_model, ngram_weight=0.1)

print("Generated text:")
print(generated_text)
