from duckduckgo_search import DDGS
from newspaper import Article
from transformers import pipeline
import warnings

# Supprimer les warnings inutiles
warnings.filterwarnings("ignore", category=UserWarning)

# === Résumeurs ===
summarizer_gpu = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)   # GPU
summarizer_cpu = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # CPU

def search_and_summarize_web(query: str, max_results: int = 3):
    """Recherche DuckDuckGo, extraction des articles, résumé multi-source"""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(keywords=query, max_results=max_results):
            urls.append(r['href'])

    sources = []
    all_text = ""
    valid_articles_found = 0  # ✅ Track number of successfully parsed articles

    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            title = article.title
            content = article.text.strip()
            if content:
                sources.append({"title": title, "url": url})
                all_text += content + "\n\n"
                valid_articles_found += 1
        except:
            continue

    if valid_articles_found == 0:
        return {
            "summary": "",
            "sources": [],
            "message": "❌ Aucun article exploitable n’a été extrait des résultats web."
        }

    # === Résumé par chunks de 1024 caractères max
    chunks = [all_text[i:i + 1024] for i in range(0, len(all_text), 1024)]
    summaries = []

    for chunk in chunks:
        input_length = len(chunk.split())
        max_len = min(250, max(32, int(input_length * 0.6)))
        min_len = min(50, max(10, int(input_length * 0.3), max_len - 1))

        try:
            summary = summarizer_gpu(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Erreur de résumé GPU: {str(e)}]")

    # === Résumé combiné sécurisé (CPU only)
    if len(summaries) > 1:
        combined = " ".join(summaries)
        combined_tokens = combined.split()

        # Sécurité : limite tokens et caractères
        if len(combined_tokens) > 900:
            combined = " ".join(combined_tokens[:900])
        if len(combined) > 3500:
            combined = combined[:3500]

        try:
            if combined.strip():
                cpu_result = summarizer_cpu(combined, max_length=250, min_length=80, do_sample=False)
                if cpu_result and isinstance(cpu_result, list) and "summary_text" in cpu_result[0]:
                    final_summary = cpu_result[0]["summary_text"]
                else:
                    final_summary = "Résumé CPU non disponible (résultat vide ou mal formé)."
            else:
                final_summary = "Aucun contenu à résumer (texte combiné vide)."
        except Exception as e:
           final_summary = "Les sources trouvées bloquent l’accès à leur contenu. Veuillez reformuler votre requête avec d’autres mots-clés ou désactiver la recherche web."




    return {
        "summary": final_summary,
        "sources": sources if sources else [{"title": "Aucune source disponible", "url": ""}]
    }

