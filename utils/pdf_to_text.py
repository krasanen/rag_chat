# utils/pdf_to_text.py
import os
from pdfminer.high_level import extract_text

def convert_pdfs_to_text(pdf_dir, txt_dir):
    os.makedirs(txt_dir, exist_ok=True)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            text = extract_text(os.path.join(pdf_dir, pdf_file))
            txt_file = os.path.join(txt_dir, pdf_file.replace('.pdf', '.txt'))
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)

if __name__ == "__main__":
    convert_pdfs_to_text('source_pdfs', 'source_txts')
