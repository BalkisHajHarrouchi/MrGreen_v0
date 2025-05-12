from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever, Document
from typing import List
import numpy as np

class RedundantFilterRetriever(BaseRetriever):
    embeddings: HuggingFaceEmbeddings
    chroma: Chroma
    k: int = 5
    similarity_threshold: float = 0.25

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_emb = self.embeddings.embed_query(query)

        mmr_docs = self.chroma.max_marginal_relevance_search(
            query=query,
            k=self.k,
            lambda_mult=0.8
        )

        scored_docs = self.chroma.similarity_search_with_score(
            query=query,
            k=self.k
        )

        filtered_docs = [
            doc for doc, score in scored_docs if score >= self.similarity_threshold
        ]

        mmr_sources = set(doc.metadata.get("source", "") for doc in mmr_docs)
        final_docs = [doc for doc in filtered_docs if doc.metadata.get("source", "") in mmr_sources]

        return final_docs if final_docs else mmr_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
