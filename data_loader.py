from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv

# ----------- Image Processing --------
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
# ----------------------------------------

load_dotenv()

EMBED_DIM = 1024
EMBED_MODEL = "mxbai-embed-large"

ollama_embedding = OllamaEmbedding(model_name=EMBED_MODEL)
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return ollama_embedding.get_text_embedding_batch(texts)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


# --- Image Processing ---
ocr_model = ocr_predictor(pretrained=True)


def load_and_chunk_image(path: str) -> list[str]:
    """
    Loads an image, extracts all text using OCR,
    and splits the text into chunks using the SAME splitter as the PDFs.
    """
    try:
        doc = DocumentFile.from_images(path)
        result = ocr_model(doc)
        full_text = result.render()
        chunks = splitter.split_text(full_text)
        return chunks

    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return []