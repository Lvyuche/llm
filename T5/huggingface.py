from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
model.to(device)  # Ensure the model is loaded to GPU if available
print(f"Device: {model.device}")

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # Enable mixed precision training for better performance
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Function to generate predictions
def generate_predictions(model, tokenizer, inputs, device):
    model.eval()
    inputs = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, axis=-1)
    return predictions.cpu().numpy()

# Sample inputs for comparison
sample_inputs = ["The food was great!", "The service was terrible.", "I love this place!", "I will never come back here again.", "The ambiance was perfect."]

# Generate predictions before fine-tuning
print("Generating predictions before fine-tuning...")
initial_predictions = generate_predictions(model, tokenizer, sample_inputs, device)
for i, prediction in enumerate(initial_predictions):
    print(f"Input: {sample_inputs[i]}")
    print(f"Prediction before fine-tuning: {prediction}")

# Evaluate the model before fine-tuning
print("Evaluating model before fine-tuning...")
initial_metrics = trainer.evaluate(eval_dataset=small_eval_dataset)
print(f"Initial accuracy: {initial_metrics['eval_accuracy']}")

# Train the model
trainer.train()

# Generate predictions after fine-tuning
print("Generating predictions after fine-tuning...")
final_predictions = generate_predictions(model, tokenizer, sample_inputs, device)
for i, prediction in enumerate(final_predictions):
    print(f"Input: {sample_inputs[i]}")
    print(f"Prediction after fine-tuning: {prediction}")

# Evaluate the model after fine-tuning
print("Evaluating model after fine-tuning...")
final_metrics = trainer.evaluate(eval_dataset=small_eval_dataset)
print(f"Final accuracy: {final_metrics['eval_accuracy']}")

# Print the performance comparison
print("Performance comparison:")
print(f"Accuracy before fine-tuning: {initial_metrics['eval_accuracy']}")
print(f"Accuracy after fine-tuning: {final_metrics['eval_accuracy']}")
