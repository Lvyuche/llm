# predict.py
import torch
from model import BiLSTMAttention
from preprocess import preprocess_text, QADataset
from torch.nn.utils.rnn import pad_sequence

def load_model(vocab_size, embeddings, embed_size=128, hidden_size=128, output_size=1):
    model = BiLSTMAttention(vocab_size, embed_size, hidden_size, output_size, embeddings)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def predict_answer(question, context, model, vocab, inv_vocab):
    question_tokens = preprocess_text(question)
    context_tokens = preprocess_text(context)
    tokens = question_tokens + context_tokens
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    inputs = torch.tensor(token_ids).unsqueeze(0)
    output = model(inputs)
    start_idx = torch.argmax(output).item()
    context_tokens = [inv_vocab[idx] for idx in inputs[0].tolist() if idx in inv_vocab]
    answer_tokens = context_tokens[start_idx:start_idx + len(preprocess_text(question))]
    answer = ' '.join(answer_tokens)
    return answer

if __name__ == "__main__":
    # Question and context
    question = "Who is the protagonist of the story?"
    
    # load model and dataset
    dataset = QADataset('data/qa_data.txt')
    model = load_model(len(dataset.vocab), dataset.embeddings)

    # predict answer
    answer = predict_answer(question, context, model, dataset.vocab, dataset.inv_vocab)
    print(f'Question: {question}\nAnswer: {answer}')
