# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocess import QADataset, preprocess_text
from model import BiLSTMAttention
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Hyperparameters
embed_size = 128
hidden_size = 128
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Custom collate function to pad sequences
def collate_fn(batch):
    questions, contexts, answers = zip(*batch)
    questions = pad_sequence(questions, batch_first=True, padding_value=0)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    return questions, contexts, answers

# Helper function to find the index of the answer in the context
def find_answer_start(context, answer, inv_vocab):
    context_tokens = [inv_vocab[idx.item()] for idx in context if idx.item() in inv_vocab]
    answer_tokens = preprocess_text(answer)
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i:i + len(answer_tokens)] == answer_tokens:
            return i
    return -1

# Function to train and evaluate the model
def train_and_evaluate(dataset, dataset_name):
    # Dataset and DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model, Loss, Optimizer
    vocab_size = len(dataset.vocab)
    embeddings = dataset.embeddings
    model = BiLSTMAttention(vocab_size, embed_size, hidden_size, vocab_size, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0
        for questions, contexts, answers in train_loader:
            questions, contexts = questions.to(device), contexts.to(device)
            optimizer.zero_grad()
            for question, context, answer in zip(questions, contexts, answers):
                question_context = torch.cat((question, context), dim=0).unsqueeze(0).to(device)
                outputs = model(question_context)
                start_idx = find_answer_start(context, answer, dataset.inv_vocab)
                if start_idx != -1 and start_idx < outputs.size(1):  # Ensure start_idx is within bounds
                    labels = torch.tensor([start_idx], dtype=torch.long).to(device)
                    loss = criterion(outputs.view(-1, vocab_size), labels)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], {dataset_name} Loss: {avg_loss:.4f}')
    
    torch.save(model.state_dict(), f'{dataset_name}_model.pth')
    return loss_history

# Load datasets
own_dataset = QADataset(['data/qa_data.txt'])
squad_dataset = QADataset(['data/squad_train.txt'])

# Training with your own dataset
loss_history_own = train_and_evaluate(own_dataset, "OwnDataset")

# Training with SQuAD dataset
loss_history_squad = train_and_evaluate(squad_dataset, "SQuAD")

# Plot the loss history
plt.plot(loss_history_own, label='Own Dataset')
plt.plot(loss_history_squad, label='SQuAD')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Comparison')
plt.show()
