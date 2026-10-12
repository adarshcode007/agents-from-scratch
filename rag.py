from typing import List
from uuid import uuid4
import math
import re
from collections import Counter

def recursive_character_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively splits text into chunks of at most chunk_size characters,
    prioritizing natural document boundaries (\n\n, \n, sentence period, space).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # Base case : if the text is smaller than the chunk size, return it
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chosen_separator = ""
    for sep in separators:
        if sep in text:
            chosen_separator = sep
            break

    if not chosen_separator:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end
        return chunks

    splits = text.split(chosen_separator)
    remaining_separators = separators[separators.index(chosen_separator) + 1 :]
    chunks = []
    current_chunk = ""

    for split in splits:
        piece = split.strip()
        if not piece:
            continue

        # Re-attach separator (INSIDE the for loop)
        candidate = f"{current_chunk}{chosen_separator}{piece}" if current_chunk else piece

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single piece itself is larger than chunk_size, recursively split it
            if len(piece) > chunk_size and remaining_separators:
                sub_chunks = recursive_character_split(
                    piece,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=remaining_separators
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = piece
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_documents_parent_child(
    text: str,
    doc_id: str,
    title: str,
    parent_size: int = 1000,
    parent_overlap: int = 100,
    child_size: int = 250,
    child_overlap: int = 30
) -> dict:
    """
    Splits a document into a two-tier hierarchy:
    1. Large Parent Chunks (rich generation context for LLM)
    2. Small Child Chunks (pinpoint vector embeddings for search)
    """
    parent_texts = recursive_character_split(
        text=text,
        chunk_size=parent_size,
        chunk_overlap=parent_overlap
    )

    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = f"{doc_id}_p{p_idx}_{uuid4().hex[:6]}"
        parents.append({
            "id": parent_id,
            "doc_id": doc_id,
            "title": title,
            "content": p_text
        })

        # Break this parent into small child chunks for vector search
        child_texts = recursive_character_split(
            text=p_text,
            chunk_size=child_size,
            chunk_overlap=child_overlap
        )

        for c_idx, c_text in enumerate(child_texts):
            child_id = f"{parent_id}_c{c_idx}"
            children.append({
                "id": child_id,
                "parent_id": parent_id,
                "doc_id": doc_id,
                "title": title,
                "content": c_text
            }) 
    return {
        "doc_id": doc_id,
        "title": title,
        "parents": parents,
        "children": children
    }


# BM25 SPARSE KEYWORD RETRIEVAL (okapi BM25)
class BM25Index:
    """
    In-memory Okapi BM25 implementation for sparse exact-keyword ranking.
    k1: Controls term frequency saturation (standard 1.5).
    b: Controls document length penalization (standard 0.75).
    """
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus # list of dicts with 'id' and 'content'
        self.doc_len=[]
        self.doc_term_freqs = []
        self.doc_freqs = Counter()
        self.avg_doc_len = 0.0
        self._build_index()
    
    def _tokenize(self, text: str) -> list[str]:
        """ Simple whitespace and punctuation tokenizer converting to lowercase tokens."""
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

    def _build_index(self):
        total_length = 0
        for doc in self.corpus:
            tokens = self._tokenize(doc["content"])
            self.doc_len.append(len(tokens))
            total_length += len(tokens)

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] += 1
        
        self.avg_doc_len = (total_length / len(self.corpus)) if self.corpus else 1.0

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Calculates BM25 score for every document in the corpus against the query."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.corpus:
            return []
        
        num_docs = len(self.corpus)
        scores = []

        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[idx]
            tf = self.doc_term_freqs[idx]

            for term in query_tokens:
                if term not in tf:
                    continue

                term_freq = tf[term]
                # Inverse Document Frequency (IDF) with smoothing
                df = self.doc_freqs.get(term,0)
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

                # Term Frequency normalization by document length
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append({
                    "id": doc["id"],
                    "parent_id": doc.get("parent_id"),
                    "doc_id": doc.get("doc_id"),
                    "title": doc.get("title"),
                    "content": doc["content"],
                    "bm25_score": round(score, 4)
                })
        # Sort descending by BM25 score
        scores.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scores[:top_k]


# Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF):
    Score(d) = sum(1 / (k+rank)) across dense and sparse lists.
    Immune to score scale differences between cosine (0-1) and BM25 (0-inf).
    """
    fused_scores = {}
    doc_meta = {}

    #1. Score dense rankings (1-indexed rank)
    for rank, doc in enumerate(dense_results, start=1):
        d_id = doc["id"]
        fused_scores[d_id] = fused_scores.get(d_id, 0.0) + (1.0 / (k+rank))
        if d_id not in doc_meta:
            doc_meta[d_id] = doc
        doc_meta[d_id]["dense_rank"] = rank
        doc_meta[d_id]["dense_score"] = doc.get("score")
    
    # 2. Score sparse rankings (1-indexed rank)
    for rank, doc in enumerate(sparse_results, start=1):
        d_id = doc["id"]
        fused_scores[d_id] = fused_scores.get(d_id, 0.0) + (1.0 / (k+rank))
        if d_id not in doc_meta:
            doc_meta[d_id] = doc
        doc_meta[d_id]["sparse_rank"] = rank
        doc_meta[d_id]["bm25_score"] = doc.get("bm25_score")

    # 3. Create scorted hybrid ranking list
    hybrid_list = []
    for d_id, rrf_score in fused_scores.items():
        entry = dict(doc_meta[d_id])
        entry["rrf_score"] = round(rrf_score, 5)
        hybrid_list.append(entry)
    
    hybrid_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return hybrid_list


            