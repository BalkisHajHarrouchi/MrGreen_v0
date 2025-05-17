
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import random
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

app = FastAPI()

# Autoriser les appels depuis le frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

# Dictionnaire de synonymes métier
SYNONYMES = {
    "programme": ["plan", "contenu", "matières", "cours"],
    "1em": ["première année", "1ère année"],
    "2em": ["deuxième année", "2ème année"],
    "3em": ["troisième année", "3ème année"],
    "matiere": ["matières", "cours"],
    "stage": ["stage d'été", "période de stage", "expérience professionnelle"],
}


# Pondération des mots-clés
KEYWORDS = {
    "1em": 3,
    "2em": 3,
    "3em": 3,
    "programme": 2,
    "matiere": 2,
    "cours": 2,
    "pfe": 2,
    "stage": 2,
    "electromecanique": 1
}

# Fonction pour extraire les mots-clés d'une question
def extract_keywords(text: str):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return set([kw for kw in KEYWORDS if kw in lemmas or any(s in lemmas for s in SYNONYMES.get(kw, []))])

# Fonction de similarité hybride : mots-clés + sémantique
def compute_hybrid_similarity(q1, q2, model):
    kw1 = extract_keywords(q1)
    kw2 = extract_keywords(q2)
    keyword_overlap = kw1.intersection(kw2)

    # Score de mots-clés (0.5 si intersection)
    keyword_score = 0.5 if keyword_overlap else 0.0

    # Score sémantique via cosine similarity
    emb1 = model.encode(q1, convert_to_tensor=True)
    emb2 = model.encode(q2, convert_to_tensor=True)
    semantic_score = float(util.cos_sim(emb1, emb2))

    # Score hybride
    return keyword_score + semantic_score

# Charger le fichier de log
def load_chat_log():
    records = []
    try:
        with open("chat_log.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except:
                    continue
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp", "question", "answer"])
    return pd.DataFrame(records)

# 🔹 Endpoint avec filtres : langue / année / mois / semaine
@app.get("/filtered-questions")
def get_filtered_questions(
    language: str = Query("all"),
    year: str = Query("all"),
    month: str = Query("all"),
    week: str = Query("all")
):
    df = load_chat_log()
    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["year"] = df["timestamp"].dt.year.astype(str)
    df["month"] = df["timestamp"].dt.month.astype(str).str.zfill(2)
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(str).str.zfill(2)

    if year != "all":
        df = df[df["year"] == year]
    if month != "all":
        df = df[df["month"] == month]
    if week != "all":
        df = df[df["week"] == week]

    if language == "fr":
        df = df[df["question"].str.contains(r"\b(le|la|est|programme|quelle|comment|des|une)\b", case=False, na=False)]
    elif language == "en":
        df = df[df["question"].str.contains(r"\b(what|is|the|how|which|when|where|program)\b", case=False, na=False)]

    return df[["timestamp", "question", "answer"]].to_dict(orient="records")

# 🔹 Endpoint sémantique avec hybrid score
@app.get("/semantic-question-stats")
def semantic_question_stats(
    language: str = Query("all"),
    year: str = Query("all"),
    month: str = Query("all"),
    week: str = Query("all")
):
    df = load_chat_log()
    if "question" not in df.columns or df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["year"] = df["timestamp"].dt.year.astype(str)
    df["month"] = df["timestamp"].dt.month.astype(str).str.zfill(2)
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(str).str.zfill(2)

    if year != "all":
        df = df[df["year"] == year]
    if month != "all":
        df = df[df["month"] == month]
    if week != "all":
        df = df[df["week"] == week]

    if language == "fr":
        df = df[df["question"].str.contains(r"\b(le|la|est|programme|quelle|comment|des|une)\b", case=False, na=False)]
    elif language == "en":
        df = df[df["question"].str.contains(r"\b(what|is|the|how|which|when|where|program)\b", case=False, na=False)]

    questions = df["question"].dropna().unique().tolist()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    clusters = []
    used = set()
    threshold = 0.95 # score minimum

    for i, q1 in enumerate(questions):
        if q1 in used:
            continue
        group = [q1]
        used.add(q1)
        for j in range(i + 1, len(questions)):
            q2 = questions[j]
            if q2 in used:
                continue
            score = compute_hybrid_similarity(q1, q2, model)
            if score >= threshold:
                group.append(q2)
                used.add(q2)
        clusters.append(group)

    output = []
    for group in clusters:
        subset = df[df["question"].isin(group)]
        output.append({
            "representative_question": random.choice(group),
            "count": len(subset),
            "all_questions": group
        })

    return sorted(output, key=lambda x: x["count"], reverse=True)

# 🔹 Endpoint par réponse (inchangé)
@app.get("/grouped-by-answer")
def grouped_by_answer():
    df = load_chat_log()
    if df.empty or "answer" not in df.columns:
        return []

    df["answer"] = df["answer"].astype(str)
    df["question"] = df["question"].astype(str)

    unique_answers = df["answer"].dropna().unique().tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_answers, show_progress_bar=False)

    if len(unique_answers) == 1:
        labels = [0]
    else:
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.35)
        labels = clustering.fit_predict(embeddings)

    output = []
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        similar_answers = [unique_answers[i] for i in indices]
        df_cluster = df[df["answer"].isin(similar_answers)]

        representative_answer = similar_answers[0]
        representative_question = random.choice(df_cluster["question"].tolist())

        output.append({
            "representative_question": representative_question,
            "representative_answer": representative_answer,
            "count": len(df_cluster),
            "all_questions": df_cluster["question"].tolist()
        })

    return sorted(output, key=lambda x: x["count"], reverse=True)
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import json
# import random
# import spacy
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from sklearn.cluster import AgglomerativeClustering
# from tqdm import tqdm

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# nlp = spacy.load("fr_core_news_sm")
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# # === Keyword Normalization ===
# def normalize_keywords(text):
#     text = text.lower()
#     replacements = {
#     # Years
#     "1ere": "1", "première": "1", "1ère": "1", "1em": "1 em", "1gc": "1 gc",
#     "2eme": "2", "deuxième": "2", "2ème": "2",
#     "3eme": "3", "troisième": "3", "3ème": "3",
#     "4ogi": "4 ogi", "5telecom": "5 telecom", "4em": "4 em",

#     # Domains
#     "electromecanique": "em", "électromécanique": "em", "em": "em",
#     "telecom": "telecom", "ogi": "ogi",

#     # Content types
#     "matiere": "programme", "matières": "programme", "cours": "programme",
#     "plan d'etude": "programme", "plan educatif": "programme", "contenu": "programme",
#     "programme": "programme",

#     # Verbs and helpers
#     "donne moi": "", "donner": "", "give me": "", "what is": "", "quel est": "", "quelle est": "",
# }

#     for k, v in replacements.items():
#         text = text.replace(k, v)
#     doc = nlp(text)
#     return set([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# def auto_label(group):
#     joined = ' '.join(group)
#     kws = normalize_keywords(joined)
#     return ', '.join(list(kws)[:2])

# def compute_hybrid_similarity_matrix(questions, keyword_weight=0.9, semantic_weight=0.1):
#     n = len(questions)
#     embeddings = model.encode(questions, convert_to_tensor=True)
#     sim_matrix = np.zeros((n, n))

#     for i in tqdm(range(n), desc="Computing hybrid matrix"):
#         for j in range(i, n):
#             kw1 = normalize_keywords(questions[i])
#             kw2 = normalize_keywords(questions[j])
#             keyword_score = len(kw1 & kw2) / max(len(kw1 | kw2), 1)
#             sem_score = float(util.cos_sim(embeddings[i], embeddings[j]))
#             score = 0.5 * keyword_score + 0.5 * sem_score
#             sim_matrix[i, j] = sim_matrix[j, i] = score
#             if i != j:
#                 print(f"--- Comparison {i} vs {j} ---")
#                 print(f"Q1: {questions[i]}\nQ2: {questions[j]}")
#                 print(f"🔍 kw1: {kw1}\n🔍 kw2: {kw2}")
#                 print(f"🔑 Keyword Similarity: {keyword_score:.2f}")
#                 print(f"🧠 Semantic Similarity: {sem_score:.2f}")
#                 print(f"🎯 Hybrid Score: {score:.2f} (Threshold = 1.00)\n")

#     return sim_matrix

# def cluster_questions(sim_matrix, questions, threshold=0.7):
#     distance_matrix = 1 - sim_matrix
#     clustering = AgglomerativeClustering(
#         n_clusters=None,
#         distance_threshold=1 - threshold,
#         metric='precomputed',
#         linkage='average'
#     )
#     labels = clustering.fit_predict(distance_matrix)
#     return labels

# @app.get("/semantic-question-stats")
# def semantic_question_stats(
#     language: str = Query("all"),
#     year: str = Query("all"),
#     month: str = Query("all"),
#     week: str = Query("all")
# ):
#     def load_chat_log():
#         records = []
#         try:
#             with open("chat_log.jsonl", "r", encoding="utf-8") as f:
#                 for line in f:
#                     try:
#                         records.append(json.loads(line.strip()))
#                     except:
#                         continue
#         except FileNotFoundError:
#             return pd.DataFrame(columns=["timestamp", "question", "answer"])
#         return pd.DataFrame(records)

#     df = load_chat_log()
#     if "question" not in df.columns or df.empty:
#         return []

#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df = df.dropna(subset=["timestamp"])

#     df["year"] = df["timestamp"].dt.year.astype(str)
#     df["month"] = df["timestamp"].dt.month.astype(str).str.zfill(2)
#     df["week"] = df["timestamp"].dt.isocalendar().week.astype(str).str.zfill(2)

#     if year != "all":
#         df = df[df["year"] == year]
#     if month != "all":
#         df = df[df["month"] == month]
#     if week != "all":
#         df = df[df["week"] == week]

#     if language == "fr":
#         df = df[df["question"].str.contains(r"\\b(le|la|est|programme|quelle|comment|des|une)\\b", case=False, na=False)]
#     elif language == "en":
#         df = df[df["question"].str.contains(r"\\b(what|is|the|how|which|when|where|program)\\b", case=False, na=False)]

#     questions = df["question"].dropna().unique().tolist()
#     sim_matrix = compute_hybrid_similarity_matrix(questions)
#     labels = cluster_questions(sim_matrix, questions, threshold=0.87)

#     df_clustered = pd.DataFrame({"question": questions, "cluster": labels})
#     output = []
#     for cluster_id, group in df_clustered.groupby("cluster"):
#         all_qs = group["question"].tolist()
#         count = len(all_qs)
#         output.append({
#             "label": auto_label(all_qs),
#             "representative_question": random.choice(all_qs),
#             "count": count,
#             "all_questions": all_qs
#         })

#     return sorted(output, key=lambda x: x["count"], reverse=True)

# @app.get("/filtered-questions")
# def get_filtered_questions(
#     language: str = Query("all"),
#     year: str = Query("all"),
#     month: str = Query("all"),
#     week: str = Query("all")
# ):
#     def load_chat_log():
#         records = []
#         try:
#             with open("chat_log.jsonl", "r", encoding="utf-8") as f:
#                 for line in f:
#                     try:
#                         records.append(json.loads(line.strip()))
#                     except:
#                         continue
#         except FileNotFoundError:
#             return pd.DataFrame(columns=["timestamp", "question", "answer"])
#         return pd.DataFrame(records)

#     df = load_chat_log()
#     if df.empty:
#         return []

#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df = df.dropna(subset=["timestamp"])

#     df["year"] = df["timestamp"].dt.year.astype(str)
#     df["month"] = df["timestamp"].dt.month.astype(str).str.zfill(2)
#     df["week"] = df["timestamp"].dt.isocalendar().week.astype(str).str.zfill(2)

#     if year != "all":
#         df = df[df["year"] == year]
#     if month != "all":
#         df = df[df["month"] == month]
#     if week != "all":
#         df = df[df["week"] == week]

#     if language == "fr":
#         df = df[df["question"].str.contains(r"\\b(le|la|est|programme|quelle|comment|des|une)\\b", case=False, na=False)]
#     elif language == "en":
#         df = df[df["question"].str.contains(r"\\b(what|is|the|how|which|when|where|program)\\b", case=False, na=False)]

#     return df[["timestamp", "question", "answer"]].to_dict(orient="records")
