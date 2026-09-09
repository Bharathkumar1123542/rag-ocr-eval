import os
import json
import csv
import ollama
import chromadb
from metrics import evaluate_query

# Filepaths
OCR_JSON_PATH = "pdfs/Assessment_doc (1).json"
CSV_OUTPUT_PATH = "results/rag_evaluation_results.csv"

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_rag_collection")


def load_and_index_ocr_json(json_path):
    """
    Reads the extracted OCR JSON file and indexes page-level text chunks 
    into ChromaDB if the database collection is currently empty.
    """
    if collection.count() > 0:
        print(f"✅ ChromaDB already contains {collection.count()} page chunks. Skipping re-indexing.")
        return

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ OCR JSON file not found: '{json_path}'. "
            f"Please place '{json_path}' in the current working directory."
        )

    print(f"📄 Reading extracted OCR data from '{json_path}'...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    # Iterate directly through the JSON list structure
    for idx, page in enumerate(data):
        page_text = page.get("text", "").strip()
        page_num = page.get("page_number", idx + 1)
        
        if page_text:
            documents.append(page_text)
            metadatas.append({
                "page_number": page_num,
                "image_path": page.get("image_path", ""),
                "line_count": page.get("line_count", 0)
            })
            ids.append(f"page_{page_num}")

    if not documents:
        raise ValueError(f"❌ No valid text content was extracted from '{json_path}'.")

    # Store text chunks into ChromaDB
    print(f"📥 Indexing {len(documents)} page chunks into ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Indexing complete!")


def run_rag_pipeline(query_id, category, query, reference_answer=None):
    """
    Queries ChromaDB for relevant PDF chunks, generates a response using
    Llama 3.2 via Ollama, and computes evaluation metrics.
    """
    # 1. Retrieve top-3 relevant page chunks from ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    retrieved_contexts = results["documents"][0] if results["documents"] else []
    context_text = "\n\n--- PAGE BREAK ---\n\n".join(retrieved_contexts)
    
    # 2. Prompt Llama 3.2 via Ollama
    prompt = f"""You are an assistant answering questions strictly based on the provided document pages.
If the context does not contain enough information to answer the question, state exactly: "Information is not present in the document."

Document Context:
{context_text}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    generated_answer = response["message"]["content"].strip()
    
    # 3. Compute Metrics
    evaluation_result = evaluate_query(
        query_id=query_id,
        category=category,
        query=query,
        generated_answer=generated_answer,
        retrieved_contexts=retrieved_contexts,
        reference_answer=reference_answer
    )
    
    return evaluation_result


def save_results_to_csv(results_list, filename=CSV_OUTPUT_PATH):
    """Saves evaluation results to CSV."""
    if not results_list:
        print("No results to save.")
        return

    fieldnames = list(results_list[0].keys())

    with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\n📊 Results successfully saved to '{filename}'")


if __name__ == "__main__":
    # Step 1: Read and Index the JSON data into ChromaDB
    load_and_index_ocr_json(OCR_JSON_PATH)

    # Step 2: Define Test Queries grounded in the PDF content
    test_questions = [
        {
            "query_id": "Q1",
            "category": "Fact-Based",
            "query": "How much total EQIP funding did New Mexico receive since 1996?",
            "reference_answer": None
        },
        {
            "query_id": "Q2",
            "category": "Analytical",
            "query": "What are the core benefits and percentage savings of the Holistic Irrigation Technology program?",
            "reference_answer": None
        },
        {
            "query_id": "Q3",
            "category": "Unanswerable",
            "query": "What was the total budget allocated for solar panel installations in 2025?",
            "reference_answer": None
        },
        {
            "query_id": "Q4",
            "category": "Fact-Based",
            "query": "Any Idea on NRCS ?",
            "reference_answer": None
        },
    ]

    print("\n=== Running RAG Evaluation Pipeline ===")
    all_results = []
    
    for item in test_questions:
        res = run_rag_pipeline(
            query_id=item["query_id"],
            category=item["category"],
            query=item["query"],
            reference_answer=item["reference_answer"]
        )
        all_results.append(res)
        
        print(f"\nID: {res['query_id']} | Category: {res['category']}")
        print(f"Query: {res['query']}")
        print(f"Answer: {res['generated_answer']}")
        print(f"Faithfulness: {res['faithfulness']} | Relevancy: {res['answer_relevancy']}")
        print(f"Composite Score: {res['composite_score']} | Penalized: {res['penalized']}")

    # Step 3: Export outputs to CSV
    save_results_to_csv(all_results, filename=CSV_OUTPUT_PATH)