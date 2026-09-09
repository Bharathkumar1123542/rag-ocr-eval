import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model globally
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def compute_contextual_precision(query, retrieved_contexts):
    if not retrieved_contexts:
        return 0.0
    q_emb = embed_model.encode(query)
    c_embs = [embed_model.encode(c) for c in retrieved_contexts]
    sims = [cosine_similarity(q_emb, c) for c in c_embs]
    return float(np.mean(sims)) if sims else 0.0

def compute_contextual_relevancy(query, retrieved_contexts):
    if not retrieved_contexts:
        return 0.0
    q_emb = embed_model.encode(query)
    c_embs = [embed_model.encode(c) for c in retrieved_contexts]
    sims = [cosine_similarity(q_emb, c) for c in c_embs]
    return float(np.max(sims)) if sims else 0.0

def compute_faithfulness(generated_answer, retrieved_contexts):
    if "information is not present" in generated_answer.lower():
        return 1.0  # Perfect faithfulness score for explicit refusals
    if not retrieved_contexts:
        return 0.0
    
    ans_emb = embed_model.encode(generated_answer)
    c_embs = [embed_model.encode(c) for c in retrieved_contexts]
    sims = [cosine_similarity(ans_emb, c) for c in c_embs]
    return float(np.max(sims)) if sims else 0.0

def compute_answer_relevancy(query, generated_answer):
    q_emb = embed_model.encode(query)
    ans_emb = embed_model.encode(generated_answer)
    return cosine_similarity(q_emb, ans_emb)

def compute_answer_correctness(generated_answer, reference_answer):
    if not reference_answer:
        return None  # Skip if no reference answer exists
    ref_emb = embed_model.encode(reference_answer)
    ans_emb = embed_model.encode(generated_answer)
    return cosine_similarity(ref_emb, ans_emb)

def evaluate_query(query_id, category, query, generated_answer, retrieved_contexts, reference_answer=None):
    is_refusal = "information is not present" in generated_answer.lower()
    
    # Base Metric Calculations
    precision = compute_contextual_precision(query, retrieved_contexts)
    relevancy_ctx = compute_contextual_relevancy(query, retrieved_contexts)
    faithfulness = compute_faithfulness(generated_answer, retrieved_contexts)
    ans_relevancy = compute_answer_relevancy(query, generated_answer)
    correctness = compute_answer_correctness(generated_answer, reference_answer)

    penalized = False
    
    if category == "Unanswerable":
        if is_refusal:
            context_recall = 0.0
            faithfulness = 1.0
            correctness_val = 1.0 if correctness is None else correctness
            composite_score = 100.0
        else:
            context_recall = 0.0
            penalized = True
            composite_score = 0.0
            correctness_val = correctness if correctness is not None else 0.0
    else:
        if is_refusal:
            context_recall = 0.0
            penalized = True
            composite_score = 15.0  # Penalty for unprompted refusal
            correctness_val = correctness if correctness is not None else 0.0
        else:
            context_recall = 1.0 if precision > 0.3 else 0.0
            correctness_val = correctness if correctness is not None else (faithfulness + ans_relevancy) / 2.0
            
            # Weighted Composite Score (scale 0-100)
            composite_score = (
                (precision * 0.2) +
                (context_recall * 0.2) +
                (faithfulness * 0.3) +
                (correctness_val * 0.3)
            ) * 100.0

    return {
        "query_id": query_id,
        "category": category,
        "query": query,
        "generated_answer": generated_answer,
        "contextual_precision": round(precision, 4),
        "contextual_recall": round(context_recall, 4),
        "contextual_relevancy": round(relevancy_ctx, 4),
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(ans_relevancy, 4),
        "answer_correctness": round(correctness_val, 4) if correctness is not None else "N/A (Custom)",
        "composite_score": round(composite_score, 2),
        "penalized": penalized
    }