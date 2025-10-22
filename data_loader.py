import os
import tempfile
import pathlib  
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

EMBED_DIM = 1536
EMBED_MODEL = "text-embedding-3-small"

embedding_model = OpenAIEmbedding(
    model=EMBED_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_extract_multimodal(pdf_path: str, source_id: str):
    """
    Extracts text and images from a PDF.
    - Text is chunked using the SentenceSplitter.
    - Images are saved to a temporary directory.
    
    Returns a dict: {"texts": list[str], "image_paths": list[str]}
    """
    
    base_temp_dir = PROJECT_ROOT / "temp_images"
    sanitized_source_id = os.path.basename(source_id).replace('.pdf', '')
    image_output_dir = base_temp_dir / sanitized_source_id
    
    os.makedirs(image_output_dir, exist_ok=True)
    
    print(f"Extracting content from {pdf_path}...")
    
    elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(image_output_dir), 
        strategy="hi_res",
    )
    
    text_elements = []
    image_paths = []
    
    for el in elements:
        if el.category == "Image":
            if hasattr(el, 'metadata') and el.metadata.image_path:
                image_paths.append(el.metadata.image_path)
            elif hasattr(el, 'image_path') and el.image_path:
                image_paths.append(el.image_path)
        elif hasattr(el, 'text') and el.text and el.text.strip():
            text_elements.append(el.text)
    
    text_chunks = []
    for t in text_elements:
        chunks = splitter.split_text(t)
        text_chunks.extend(chunks)
    
    print(f"Extracted {len(text_chunks)} text chunks and {len(image_paths)} images.")
    
    return {"texts": text_chunks, "image_paths": image_paths}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts using OpenAI's text-embedding-3-small model.
    Returns a list of embedding vectors (each is a list of 1536 floats).
    """
    if not texts:
        return []
    
    print(f"Embedding {len(texts)} text(s) using {EMBED_MODEL}...")
    
    try:
        embeddings = embedding_model.get_text_embedding_batch(texts)
        return embeddings
    except Exception as e:
        print(f"Error during embedding: {e}")
        raise