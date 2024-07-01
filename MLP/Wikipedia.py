import requests
import json

def get_wikipedia_articles(keywords, num_articles_per_keyword=10):
    url = "https://en.wikipedia.org/w/api.php"
    articles = []

    for keyword in keywords:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': keyword,
            'srlimit': num_articles_per_keyword
        }

        response = requests.get(url, params=params)
        data = response.json()

        for article in data['query']['search']:
            page_id = article['pageid']
            title = article['title']

            article_params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'explaintext': True,
                'pageids': page_id
            }

            article_response = requests.get(url, params=article_params)
            article_data = article_response.json()
            extract = article_data['query']['pages'][str(page_id)]['extract']

            articles.append({
                'title': title,
                'content': extract
            })

    return articles

# 定义关键词
keywords = ["machine learning", "deep learning", "neural network", "natural language processing", "artificial intelligence"]

# 获取Wikipedia文章
articles = get_wikipedia_articles(keywords)

# 保存到文件
with open('wikipedia_articles.json', 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print("Downloaded articles saved to wikipedia_articles.json")
