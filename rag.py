from typing import List
from uuid import uuid4
import math
import re
from collections import Counter
import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

rag_client = Groq()

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


# QUERY TRANSFORMATION / MULTI-QUERY EXPANSION
def expand_query_multi_angle(query: str, num_queries: int = 3) -> list[str]:
    """
    Query Transformation:
    Takes an ambiguous or conversational user query and genenrates 3 distinct,
    techincally-targeted search variations from different perspectives.
    """
    prompt = f"""
        You are an expert search retrieval engineer.
        Given the user's search query, generate {num_queries} distinct,
        highly specific search queries to retrieve relevant technical documentation and runbooks.
        Focus on:
        1. Exact techinal keywords and error patterns.
        2. Conceptual synonyms and underlying system causes.
        3. Diagnostic and operational troubleshooting terms.

        Return ONLY a valid JSON array of {num_queries} strings. Example: {{"queries": ["query 1", "query 2", "query 3"]}}

        User Query: "{query}"
    """
    try:
        completion = rag_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages = [
                {"role": "system", "content": "You are a query expansion engine. Ouptut only a JSON array of strings."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        data = json.loads(completion.choices[0].message.content)
        queries = data.get("queries",[])
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries[:num_queries]]
        return [query]
    except Exception:
        # Graceful fallback: return original query if expansion fails
        return [query]


# CROSS-ENCODER RE-RANKER
def rerank_candidates_cross_encoder(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Cross-Encoder Re-ranking:
    Evaluates (Query, Candidate) pairs with full cross-attention to score
    semantic relevance on a continuous scale (0.0 to 1.0)
    """
    if not candidates:
        return []
    
    # Prepare candidate snippets for cross-attention evaluation
    candidate_prompts = []
    for idx, c in enumerate(candidates):
        snippet = c.get("matched_child_snippet") or c.get("parent_content","")[:300]
        candidate_prompts.append(f"[{idx}] {snippet}")

    candidates_text = "\n\n".join(candidate_prompts)

    rerank_prompt = f"""
        You are a Cross-Encoder Re-ranking model.
        Evaluate how well candidate document directly answers or provides context for the query.
        Assign a relevance score from 0.00 (completely irrelevant) to 1.00 (perfect direct answer).

        Query: "{query}"

        Candidates: {candidates_text}

        Output ONLY a JSON object with key "rankings" containing a list of objects with keys "index" (integer) and "relevance_score" (float 0.0 to 1.0).
        Example: {{"rankings": [{{"index": 0, "relevance_score": 0.95}}, {{"index": 1, "relevance_score": 0.32}}]}}
    """

    try:
        completion = rag_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a strict Cross-Encoder ranking engine. Ouput only JSON."},
                {"role": "user", "content": rerank_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        raw_res = json.loads(completion.choices[0].message.content)
        scores_list = raw_res.get("rankings",[]) if isinstance(raw_res, dict) else raw_res

        # Map cross-encoders scores back to candidate dicts
        for item in scores_list:
            idx = int(item.get("index"))
            score = float(item.get("relevance_score", 0.0))
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                candidates[idx]["cross_encoder_score"] = round(score, 4)

        # Sort by Cross-Encoder score descending
        reranked = sorted(candidates, key=lambda x: x.get("cross_encoder_score",0.0),reverse=True)
        return reranked[:top_k]
    except Exception:
        # Fallback: if re-ranking fails, return original candidates
        return candidates[:top_k]


def advanced_rag_search(query: str, search_fn, top_k: int = 3, num_expanded_queries: int = 3) -> dict:
    """
    End-to-End Advanced RAG Pipeline:
    1. Query Transformation (Multi-Query Expansion)
    2. Parallel / Multi-angle Hybrid Search (Dense + BM25 + RRF)
    3. Candidate pooling & deduplication across expanded queries
    4. Cross-Encoder Re-rankinng to pick top_k true winners 
    """
    # 1. Expand query into multiple angles
    expanded_queries = expand_query_multi_angle(query, num_queries=num_expanded_queries)
    if query not in expanded_queries:
        expanded_queries.insert(0, query)
    
    # 2. Retrieve candidates for each variation using hybrid search
    candidate_pool = {}
    for q in expanded_queries:
        matches = search_fn(query=q, top_k=5)
        for m in matches:
            p_id = m["parent_id"]
            # Keep highest initial score if seen across multiple sub-queries
            if p_id not in candidate_pool or m.get("rrf_score", 0) > candidate_pool[p_id].get("rrf_score", 0):
                candidate_pool[p_id] = m
    pooled_candidates = list(candidate_pool.values())

    # 3. Re-rank pooled candidates with Cross-Encoder
    final_ranked = rerank_candidates_cross_encoder(
        query=query,
        candidates=pooled_candidates,
        top_k=top_k
    )

    return {
        "original_query": query,
        "expanded_queries": expanded_queries,
        "total_candidates_pooled": len(pooled_candidates),
        "results": final_ranked
    }