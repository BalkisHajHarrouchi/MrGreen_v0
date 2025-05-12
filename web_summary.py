from duckduckgo_search import DDGS
from newspaper import Article
from transformers import pipeline
import textwrap
import warnings

# === Supprime les warnings inutiles (longueur, GPU, etc.)
warnings.filterwarnings("ignore", category=UserWarning)

# === Résumeur HuggingFace (léger mais efficace)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def search_web(query, max_results=3):
    """Recherche sur le web avec DuckDuckGo et retourne les URLs"""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(keywords=query, max_results=max_results):  # 🛠 keywords= obligatoire
            urls.append(r['href'])
    return urls

def extract_full_text(url):
    """Extrait le texte complet de la page via newspaper3k"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text.strip()
    except:
        return "", ""

def summarize_text(text, max_tokens=1024):
    """Divise le texte et génère un résumé global"""
    chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    summaries = []

    for chunk in chunks:
        input_length = len(chunk.split())
        max_len = min(250, max(32, int(input_length * 0.6)))
        min_len = min(50, max(10, int(input_length * 0.3), max_len - 1))

        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    if len(summaries) > 1:
        return summarize_text(" ".join(summaries))
    else:
        return summaries[0]

# === Script principal ===
if __name__ == "__main__":
    print("✅ Script started.")
    question = input("Pose ta question : ")
    print("\n🔍 Recherche en cours...\n")
    urls = search_web(question, max_results=3)

    all_content = ""
    for url in urls:
        title, content = extract_full_text(url)
        if content:
            print(f"🔗 Source: {url}")
            print(f"📌 Titre: {title}")
            print("\n📝 Extrait de l'article :\n")
            excerpt = " ".join(content.split()[:150])
            print(textwrap.fill(excerpt, width=100))
            print("\n" + "-" * 100 + "\n")
            all_content += content + "\n\n"
        else:
            print(f"❌ Impossible d'extraire le contenu de : {url}\n")

    if all_content:
        print("\n🧠 Résumé des 3 sources :\n")
        summary = summarize_text(all_content)
        print(textwrap.fill(summary, width=100))
    else:
        print("⚠️ Aucun contenu à résumer.")
