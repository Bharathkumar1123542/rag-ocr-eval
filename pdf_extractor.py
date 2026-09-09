import fitz  # PyMuPDF pip install pymupdf
import json

def extract_pdf_default(pdf_path: str, output_txt: str = "extracted_text.txt"):
    """
    Extracts text page by page using default PyMuPDF settings with layout sorting enabled.
    """
    doc = fitz.open(pdf_path)
    print(f"--- Loaded PDF: {pdf_path} ---")
    print(f"Total Pages: {len(doc)}\n")

    full_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # sort=True sorts output by vertical then horizontal coordinates
        # This prevents text from jumping across columns out of order.
        page_text = page.get_text("text", sort=True)
        
        # Header banner for readability
        page_header = f"\n=== PAGE {page_num + 1} OF {len(doc)} ===\n"
        print(page_header)
        
        # Print first 300 characters to stdout to check visually
        preview = page_text[:300].replace('\n', ' ')
        print(f"Preview: {preview}...\n")
        
        full_text.append(f"{page_header}\n{page_text}")

    # Save complete extracted text to disk
    combined_content = "\n".join(full_text)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(combined_content)
        
    print(f"✅ Extraction complete! Full content written to: {output_txt}")

if __name__ == "__main__":
    # Replace with your local PDF filename
    pdf_filename = "/Users/venkatesh.manohar022/Documents/test/pdfs/Assessment_doc (1).pdf" 
    extract_pdf_default(pdf_filename)