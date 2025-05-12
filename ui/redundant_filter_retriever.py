from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):
    embeddings: HuggingFaceEmbeddings
    chroma: Chroma
    k: int = 5

    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
            k=self.k
        )

    async def aget_relevant_documents(self):
        return []

