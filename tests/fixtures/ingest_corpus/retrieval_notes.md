# Retrieval notes

We rank documents by how close their embeddings are to the query. A nearest neighbour
search over vector representations finds passages with similar meaning even when the
wording differs, and a reranker then reorders the shortlist.
