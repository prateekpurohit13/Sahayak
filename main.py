import logging
import base64
import os 
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime
from openai import OpenAI
from data_loader import load_and_extract_multimodal, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult, RAGUpsertResult 

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


def encode_image(image_path: str) -> str:
    """Encodes an image file into a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return ""
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return ""


def get_image_mime_type(image_path: str) -> str:
    """Determines the MIME type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/png')


def _summarize_image(image_path: str) -> str:
    """
    Summarizes an image using the OpenAI GPT-4o model.
    Focuses on both visual content and any text present in the image.
    """
    print(f"Sending image {image_path} to OpenAI for summarization...")
    base64_image = encode_image(image_path)
    if not base64_image:
        return ""
    
    # Get the correct MIME type
    mime_type = get_image_mime_type(image_path)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Analyze this image and provide a detailed but concise summary. Include:
1. Main visual elements and what they depict
2. Any text visible in the image (transcribe it exactly)
3. Key information or concepts being conveyed
4. Context or purpose of the image

Keep the summary clear and structured for educational purposes."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        summary = response.choices[0].message.content.strip()
        # Add a prefix to distinguish it from normal text
        return f"[Image Content] {summary}"
    except Exception as e:
        logging.error(f"Error summarizing image with OpenAI: {e}")
        return ""


def _call_llm(prompt_content: str) -> str:
    """Calls the OpenAI API for text generation."""
    print("Sending request to OpenAI API...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are Sahayak, a compassionate and expert AI teaching assistant for schools in rural India."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return "Sorry, I encountered an error while processing your request."


@inngest_client.create_function(
    fn_id="RAG: Ingest Document", 
    trigger=inngest.TriggerEvent(event="rag/ingest_document"), 
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_document(ctx: inngest.Context):
    
    def _load(ctx: inngest.Context) -> dict:
        pdf_path = ctx.event.data.get("pdf_path")
        image_path = ctx.event.data.get("image_path")
        
        source_id = ctx.event.data.get("source_id")
        if not source_id:
            if pdf_path:
                source_id = pdf_path
            elif image_path:
                source_id = image_path
            else:
                raise ValueError("Either 'pdf_path' or 'image_path' must be provided.")

        extracted_texts = []
        extracted_image_paths = []

        if pdf_path:
            print(f"Processing PDF: {pdf_path}")
            extracted_data = load_and_extract_multimodal(pdf_path, source_id)
            extracted_texts.extend(extracted_data["texts"])
            extracted_image_paths.extend(extracted_data["image_paths"])
        elif image_path:
            print(f"Processing standalone image: {image_path}")
            # For standalone images, the path *is* the image path.
            # It must be accessible by the server.
            extracted_image_paths.append(image_path)
        else:
            raise ValueError("Event data must contain either 'pdf_path' or 'image_path'.")
            
        return {
            "texts": extracted_texts,
            "image_paths": extracted_image_paths,
            "source_id": source_id
        }

    def _upsert(extracted_data: dict, language: str) -> RAGUpsertResult:
        text_chunks = extracted_data["texts"]
        image_paths = extracted_data["image_paths"]
        source_id = extracted_data["source_id"]
        
        all_texts_to_embed = []
        all_payloads = []
        
        # Process text chunks
        for i, chunk in enumerate(text_chunks):
            all_texts_to_embed.append(chunk)
            all_payloads.append({
                "source": source_id, 
                "text": chunk, 
                "language": language,
                "media_type": "text"
            })

        # Process images - summarize and embed
        for i, img_path in enumerate(image_paths):
            print(f"Summarizing image {i+1}/{len(image_paths)}: {img_path}")
            summary = _summarize_image(img_path) 
            
            if summary:
                all_texts_to_embed.append(summary)
                all_payloads.append({
                    "source": source_id,
                    "text": summary,
                    "language": language,
                    "media_type": "image_summary",
                    "original_image_path": img_path 
                })
            else:
                logging.warning(f"Failed to summarize image: {img_path}")
        
        if not all_texts_to_embed:
            logging.warning("No text or image content to embed.")
            return RAGUpsertResult(ingested=0)

        print(f"Embedding {len(all_texts_to_embed)} total items...")
        vecs = embed_texts(all_texts_to_embed)
        
        # Generate unique IDs for each chunk/image
        ids = []
        for i, payload in enumerate(all_payloads):
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}:{payload['media_type']}")))
        
        QdrantStorage().upsert(ids, vecs, all_payloads)
        print(f"Successfully upserted {len(ids)} items to vector database.")
        return RAGUpsertResult(ingested=len(ids))
    
    language = ctx.event.data.get("language", "en")
    
    extracted_data = await ctx.step.run(
        "load-and-extract-multimodal", 
        lambda: _load(ctx), 
        output_type=dict
    )
    
    ingested = await ctx.step.run(
        "summarize-embed-upsert-multimodal", 
        lambda: _upsert(extracted_data, language),
        output_type=RAGUpsertResult
    )
    
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", 
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai") 
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5, language: str = None) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k, language=language)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    language = ctx.event.data.get("language", "en")

    found = await ctx.step.run(
        "embed-and-search", 
        lambda: _search(question, top_k, language),
        output_type=RAGSearchResult
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    
    prompt = f"""You are answering a question for a student in rural India. Use the provided context to answer accurately and helpfully.

**Context:**
{context_block}

**Question:**
{question}

**Instructions:**
1. First, provide a clear and accurate explanation based strictly on the context above.
2. Then, create a simple analogy or example from rural Indian village life (farming, cooking, nature) to make it relatable.
3. Respond entirely in {language} language.

**Answer:**"""

    answer = await ctx.step.run("llm-answer", lambda: _call_llm(prompt))
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_document, rag_query_pdf_ai])