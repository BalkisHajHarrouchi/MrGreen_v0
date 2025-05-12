from pdfplumber import open as pdf_open
from PIL import Image
from docx import Document
from pdf2image import convert_from_path
import pytesseract, csv

def ocr_pdf(path): return "\n".join(pytesseract.image_to_string(p) for p in convert_from_path(path))
def extract_text_from_pdf(path):
    try:
        with pdf_open(path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text if text.strip() else ocr_pdf(path)
    except Exception as e: return f"❌ PDF error: {e}"
def extract_text_from_docx(path): return "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
def extract_text_from_image(path): return pytesseract.image_to_string(Image.open(path))
def extract_text_from_txt(path): return open(path, "r", encoding="utf-8").read()
def extract_text_from_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return "\n".join(", ".join(row) for row in csv.reader(f))
