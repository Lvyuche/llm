from datasets import load_dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the local JSON file
dataset = load_dataset('json', data_files='qa_dataset.json', field='data')

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Define the tokenization function
def preprocess_function(examples):
    questions = examples["question"]
    contexts = examples["context"]
    inputs = ["question: " + q + " context: " + c for q, c in zip(questions, contexts)]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process the entire dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split the dataset: 80% for training, 20% for validation
split_datasets = tokenized_datasets['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets['train']
validation_dataset = split_datasets['test']

# Set training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)

# Function to generate answers before fine-tuning
def generate_answer_before(question, context, max_length=30):
    inputs = tokenizer("question: " + question + " context: " + context, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        min_length=5, 
        num_beams=4, 
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Generate answers before fine-tuning
question = "What did Jim sell to buy Della's gift?"
context = "Jim took out a small, carefully wrapped package from his overcoat and handed it to Della. 'Open it,' he said softly. Inside, Della found a set of beautiful combs for her hair, something she had always wanted but never thought she could afford. Jim smiled and said, 'I sold my watch to buy them for you."
print("Before fine-tuning:")
print(generate_answer_before(question, context, max_length=30))  # Output length is 30

# Start training
trainer.train()

# Function to generate answers after fine-tuning
def generate_answer_after(question, context, max_length=30):
    inputs = tokenizer("question: " + question + " context: " + context, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        min_length=5, 
        num_beams=4, 
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Generate answers after fine-tuning
print("After fine-tuning:")
print(generate_answer_after(question, context, max_length=30))  # Output length is 30
