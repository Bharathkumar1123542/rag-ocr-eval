import os
import json
import logging
import numpy as np
import pymupdf as fitz  # PyMuPDF
from paddleocr import PaddleOCR

# Suppress background debug logs from PaddleOCR
logging.getLogger("ppocr").setLevel(logging.ERROR)

# =========================================================
# PADDLEOCR WRAPPER INITIALIZATION
# Uses updated API parameters compatible with modern PaddleOCR releases
# =========================================================
ocr = PaddleOCR(
    use_textline_orientation=True,  # Orientation / angle classifier
    lang='en',
    ocr_version='PP-OCRv4',          # Pipeline version
    text_det_thresh=0.3,             # Detect faint/thin scanned text
    text_det_box_thresh=0.5,         # Filter out noisy background boxes
    text_det_unclip_ratio=1.6,       # Prevent boundary text clipping
)

def parse_bbox_coords(bbox):
    """
    Safely extracts top_y and left_x from any PaddleOCR bbox structure:
    - 2D vertex list/array: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    - 1D box list/array:    [x_min, y_min, x_max, y_max]
    - NumPy arrays or nested lists
    """
    try:
        if bbox is None:
            return 0.0, 0.0

        # Convert numpy array to python list if applicable
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()

        if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
            return 0.0, 0.0

        first_elem = bbox[0]

        # Case 1: 2D array [[x1, y1], [x2, y2], ...]
        if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) >= 2:
            left_x = float(first_elem[0])
            top_y = float(first_elem[1])
            return top_y, left_x

        # Case 2: 1D array [x_min, y_min, x_max, y_max]
        if len(bbox) >= 2:
            left_x = float(bbox[0])
            top_y = float(bbox[1])
            return top_y, left_x

    except (IndexError, TypeError, ValueError):
        pass

    return 0.0, 0.0


def run_paddle_ocr_pipeline(pdf_path: str, output_json: str = None):
    """
    1. Renders PDF pages into images in a unique temp folder.
    2. Runs PaddleOCR over saved images using .predict().
    3. Safely parses bounding box coordinates regardless of NumPy or list array shape.
    4. Sorts extracted lines spatially (top-to-bottom, left-to-right).
    5. Saves output as a JSON file named after the input PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # Extract base filename without extension (e.g., "Assessment_doc (1)")
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_dir = os.path.dirname(pdf_path) or "."

    # Dynamically set output JSON path using the PDF name if not explicitly provided
    if output_json is None:
        output_json = os.path.join(pdf_dir, f"{pdf_base_name}.json")

    # Create unique temp image directory
    temp_dir = os.path.join(pdf_dir, f"temp_images_{pdf_base_name}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"--- Created temporary image directory: {temp_dir} ---")

    doc = fitz.open(pdf_path)
    print(f"--- Processing PDF: {pdf_path} ({len(doc)} pages) ---")

    # STEP 1: Render and dump PDF pages to 300 DPI images
    image_paths = []
    print("\n[Step 1/2] Converting PDF pages to 300 DPI images...")
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_name = f"page_{page_num + 1}.png"
        img_path = os.path.join(temp_dir, img_name)
        pix.save(img_path)
        image_paths.append((page_num + 1, img_path))
        print(f" Saved: {img_path}")

    # STEP 2: Extract text using PaddleOCR
    print("\n[Step 2/2] Running PaddleOCR extraction on saved images...")
    extracted_pages = []

    for page_num, img_path in image_paths:
        print(f"Extracting text from Page {page_num}/{len(doc)} ({os.path.basename(img_path)})...")

        result = ocr.predict(img_path)
        page_lines = []

        if result and len(result) > 0:
            res_data = result[0]

            # Case 1: Dictionary-style output (PaddleOCR 3.x / modern API)
            if isinstance(res_data, dict):
                boxes = res_data["rec_boxes"] if "rec_boxes" in res_data and res_data["rec_boxes"] is not None else res_data.get("boxes", [])
                texts = res_data["rec_texts"] if "rec_texts" in res_data and res_data["rec_texts"] is not None else res_data.get("txts", [])
                scores = res_data["rec_scores"] if "rec_scores" in res_data and res_data["rec_scores"] is not None else res_data.get("scores", [])

                boxes = [] if boxes is None else boxes
                texts = [] if texts is None else texts
                scores = [] if scores is None else scores

                for bbox, text_content, confidence in zip(boxes, texts, scores):
                    if confidence > 0.5 and str(text_content).strip():
                        top_y, left_x = parse_bbox_coords(bbox)
                        page_lines.append({
                            "top_y": top_y,
                            "left_x": left_x,
                            "text": str(text_content).strip()
                        })

            # Case 2: Tuple/List-style output (PaddleOCR 2.x legacy structure)
            elif isinstance(res_data, list):
                for item in res_data:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        bbox, text_tuple = item
                        if isinstance(text_tuple, (list, tuple)) and len(text_tuple) == 2:
                            text_content, confidence = text_tuple
                            if confidence > 0.5 and str(text_content).strip():
                                top_y, left_x = parse_bbox_coords(bbox)
                                page_lines.append({
                                    "top_y": top_y,
                                    "left_x": left_x,
                                    "text": str(text_content).strip()
                                })

            # Sort lines spatially (top-to-bottom using 25px row buckets, then left-to-right)
            page_lines.sort(key=lambda item: (item["top_y"] // 25, item["left_x"]))

        combined_page_text = "\n".join([item["text"] for item in page_lines])

        extracted_pages.append({
            "page_number": page_num,
            "image_path": img_path,
            "text": combined_page_text,
            "line_count": len(page_lines)
        })

    # Save to JSON named after the original PDF file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_pages, f, indent=2, ensure_ascii=False)

    print(f"\n✅ PaddleOCR extraction complete! Results saved to: {output_json}")
    return output_json


if __name__ == "__main__":
    pdf_filename = "/Users/venkatesh.manohar022/Documents/test/pdfs/Assessment_doc (1).pdf"
    run_paddle_ocr_pipeline(pdf_filename)