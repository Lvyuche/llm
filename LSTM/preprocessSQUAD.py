# preprocess_squad.py
import json
import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())  # Convert to lowercase to ensure consistency
    return tokens

def load_squad_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    data = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if len(qa['answers']) > 0:
                    answer = qa['answers'][0]['text']
                else:
                    answer = ""
                data.append((question, context, answer))
    return data

def save_preprocessed_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for question, context, answer in data:
            f.write(f"{question}\t{context}\t{answer}\n")

if __name__ == "__main__":
    squad_train_data = load_squad_data('data/train-v2.0.json')  # Update with correct SQuAD file path
    squad_dev_data = load_squad_data('data/dev-v2.0.json')  # Update with correct SQuAD file path

    save_preprocessed_data(squad_train_data, 'data/squad_train.txt')
    save_preprocessed_data(squad_dev_data, 'data/squad_dev.txt')
