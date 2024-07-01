import re
from collections import Counter
import matplotlib.pyplot as plt

def analyze_word_frequencies(filepath):
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    # Count word frequencies
    word_counts = Counter(re.findall(r'\w+|[^\w\s]', text))

    # Get the 50 most common words
    most_common_words = word_counts.most_common(50)

    # Separate words and counts for plotting
    words, counts = zip(*most_common_words)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(words, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 50 Most Common Words')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    # Replace 'data/processed_articles.txt' with your actual file path
    filepath = 'data/processed_articles.txt'
    analyze_word_frequencies(filepath)
