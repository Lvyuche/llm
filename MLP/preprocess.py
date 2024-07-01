import json
import re

def clean_text(text):
    # Remove special symbols and formatting characters
    # Remove LaTeX or other special formatted text within curly braces
    text = re.sub(r'\{[^}]*\}', '', text)
    # Remove special characters and numbers (retain only letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_wikipedia_articles(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        articles = json.load(infile)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for article in articles:
            content = article['content']
            cleaned_content = clean_text(content)
            outfile.write(cleaned_content + "\n")

# 预处理文章
preprocess_wikipedia_articles('wikipedia_articles.json', 'processed_articles.txt')

print("Processed articles saved to processed_articles.txt")
