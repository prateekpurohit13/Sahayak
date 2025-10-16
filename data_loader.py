from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

load_dotenv()

EMBED_DIM = 1024
EMBED_MODEL = "mxbai-embed-large"

ollama_embedding = OllamaEmbedding(model_name=EMBED_MODEL)
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return ollama_embedding.get_text_embedding_batch(texts)