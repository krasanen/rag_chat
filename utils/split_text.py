# utils/split_text.py
import os
from transformers import GPT2Tokenizer

def split_text(text, max_length=500):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def process_texts(txt_dir, chunk_dir):
    os.makedirs(chunk_dir, exist_ok=True)
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = split_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_file = f"{txt_file.replace('.txt', '')}_chunk_{idx}.txt"
                with open(os.path.join(chunk_dir, chunk_file), 'w', encoding='utf-8') as cf:
                    cf.write(chunk)

if __name__ == "__main__":
    process_texts('source_txts', 'source_chunks')
