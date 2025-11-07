import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime
import requests
import json
import os
import tempfile
import openai
import whisper
from data_loader import (
    load_and_chunk_pdf, 
    embed_texts, 
    load_and_chunk_image
)
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here")

LANGUAGE_KEYWORDS = {
    "hi": ["क", "ख", "ग", "घ", "च", "छ", "है", "की", "का", "में", "से", "को", "और", "यह", "वह"],
    "en": ["the", "is", "are", "what", "how", "why", "when", "where", "explain", "tell", "describe"]
}

LANGUAGE_TRIGGERS = {
    "hi": ["hindi", "हिंदी", "हिन्दी", "in hindi"],
    "en": ["english", "in english"]
}

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

try:
    whisper_model = whisper.load_model("base")
    logging.info("Local Whisper model 'base' loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load local Whisper model: {e}")
    whisper_model = None

def _detect_language(text: str) -> str:
    if not text:
        return DEFAULT_LANGUAGE  
    text_lower = text.lower()
    for lang, triggers in LANGUAGE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in text_lower:
                logging.info(f"Language explicitly requested: {lang} (trigger: '{trigger}')")
                return lang
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "hi"
    
    hindi_matches = sum(1 for keyword in LANGUAGE_KEYWORDS["hi"] if keyword in text)
    english_matches = sum(1 for keyword in LANGUAGE_KEYWORDS["en"] if keyword in text_lower)
    
    if hindi_matches > english_matches:
        return "hi"
    elif english_matches > 0:
        return "en"
    
    return DEFAULT_LANGUAGE

MAX_TELEGRAM_MESSAGE_LENGTH = 4000


def _chunk_text(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    chunks = []
    current = []
    current_len = 0
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            paragraph = ""
        if current_len + len(paragraph) + 1 > limit and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    if not chunks:
        return [text[:limit]]
    return chunks


def _send_telegram_message(chat_id: str, text: str):
    if not text:
        text = ""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for chunk in _chunk_text(text):
        payload = {"chat_id": chat_id, "text": chunk}
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send message to Telegram: {e}")
            if payload.get("text") != chunk.replace("\n", " "):
                fallback_payload = {"chat_id": chat_id, "text": chunk.replace("\n", " ")}
                try:
                    response = requests.post(url, json=fallback_payload, timeout=15)
                    response.raise_for_status()
                except requests.exceptions.RequestException as inner_e:
                    logging.error(f"Fallback send failed: {inner_e}")
                    raise
            else:
                raise

def _download_telegram_file(file_id: str) -> str:
    # Downloads a file from Telegram and saves it to a temporary path.
    get_path_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
    path_response = requests.get(get_path_url).json()
    if not path_response.get("ok"):
        raise Exception("Failed to get file path from Telegram")
    
    file_path = path_response["result"]["file_path"]
    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    
    file_response = requests.get(download_url)
    file_response.raise_for_status()
    _, ext = os.path.splitext(file_path)
    fd, temp_path = tempfile.mkstemp(suffix=ext or "")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_response.content)
    except Exception:
        os.remove(temp_path)
        raise
    return temp_path

def _call_ollama(prompt_content: str) -> str:
    # Call Ollama API and return the response (fallback if OpenAI not available)
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3:8b",
        "prompt": prompt_content,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        return json.loads(response.text)["response"].strip()
    except requests.exceptions.Timeout:
        logging.error("Ollama request timed out")
        raise Exception("Ollama is taking too long to respond. Please try again.")
    except requests.exceptions.ConnectionError:
        logging.error("Cannot connect to Ollama")
        raise Exception("Cannot connect to Ollama. Make sure it's running on localhost:11434")
    except Exception as e:
        logging.error(f"Ollama error: {e}")
        raise Exception(f"Error calling Ollama: {str(e)}")


def _call_openai(prompt_content: str) -> str:
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not configured")
    
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            if content:
                return content.strip()
        
        raise Exception("Unexpected response format from OpenAI")
        
    except requests.exceptions.Timeout:
        logging.error("OpenAI request timed out")
        raise Exception("OpenAI API is taking too long to respond. Please try again.")
    except requests.exceptions.RequestException as e:
        logging.error(f"OpenAI API error: {e}")
        raise Exception(f"Error calling OpenAI API: {str(e)}")


def _call_llm(prompt_content: str) -> str:
    if USE_OPENAI:
        try:
            logging.info("Using OpenAI API")
            return _call_openai(prompt_content)
        except Exception as e:
            logging.warning(f"OpenAI failed, falling back to Ollama: {e}")

    logging.info("Using Ollama (local)")
    return _call_ollama(prompt_content)

def _transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using the LOCAL Whisper model.
    """
    if not whisper_model:
        logging.error("Local Whisper model is not loaded. Cannot transcribe audio.")
        raise Exception("Local Whisper model is not loaded. Cannot transcribe audio.")

    try:
        # The transcribe() function handles everything
        result = whisper_model.transcribe(file_path)
        
        transcribed_text = result.get("text", "")
        if transcribed_text:
            return transcribed_text.strip()
        
        logging.warning("Local transcription returned no text.")
        raise Exception("Transcription returned no text.")
        
    except Exception as e:
        logging.error(f"Local transcription failed: {e}")
        raise Exception(f"Error transcribing audio locally: {str(e)}")

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    pdf_path = ctx.event.data["pdf_path"]
    language = ctx.event.data.get("language", "en")
    source_id = ctx.event.data.get("source_id", pdf_path)
    chunks = load_and_chunk_pdf(pdf_path)
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i], "language": language} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    return {"ingested": len(chunks)}


@inngest_client.create_function(
    fn_id="RAG: Ingest Image",
    trigger=inngest.TriggerEvent(event="rag/ingest_image"),
)
async def rag_ingest_image(ctx: inngest.Context):
    image_path = ctx.event.data["image_path"]
    language = ctx.event.data.get("language", "en")
    source_id = ctx.event.data.get("source_id", image_path)
    chunks = load_and_chunk_image(image_path)
    if not chunks:
        return {"ingested": 0, "status": "No text found"}
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i], "language": language} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    return {"ingested": len(chunks)}

@inngest_client.create_function(
    fn_id="RAG: Query AI",
    trigger=inngest.TriggerEvent(event="rag/query_ai")
)
async def rag_query_ai(ctx: inngest.Context):
    user_id = ctx.event.data["user_id"]
    question = ctx.event.data["question"]
    language = ctx.event.data.get("language", "en")
    top_k = int(ctx.event.data.get("top_k", 5))

    try:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k, language=language)
        
        if not found["contexts"]:
            await ctx.step.run("send-no-context-reply", 
                lambda: _send_telegram_message(user_id, 
                    "I couldn't find relevant information to answer your question. Please make sure educational content has been uploaded to the database. 📚"))
            return {"status": "no_context_found"}
        
        context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
        prompt = f"""
System Prompt: You are "Sahayak," a compassionate and expert AI teaching assistant for schools in rural India. Your primary goal is to make learning intuitive and relatable. Your tone should be encouraging, simple, and clear.
Your task is to answer the user's question by following these two steps precisely:
1.  **The Explanation:** First, provide a simple and factually correct explanation that directly answers the user's question. This explanation MUST be based strictly on the information within the "Provided Context" below. Do not add any information that is not in the context for this part.
2.  **The Relatable Example:** Second, and most importantly, create a vivid and simple analogy or example that connects the main point of the explanation to the daily life of a student in a rural Indian village. Use familiar elements like farming (kheti), cooking (rasoi), local nature (prakriti), a village market (bazaar), or common household items.
The entire response must be in the {language} language.
---
Provided Context:\n{context_block}\n---
User Question:\n{question}\n---
Answer:
"""
        answer = await ctx.step.run("llm-answer", lambda: _call_llm(prompt))
        await ctx.step.run("send-reply", lambda: _send_telegram_message(user_id, answer))
        return {"status": "success", "answer_sent": True}
    
    except Exception as e:
        logging.error(f"Error in rag_query_ai: {e}")
        await ctx.step.run("send-error-reply", 
            lambda: _send_telegram_message(user_id, 
                "Sorry, I encountered an error while processing your question. Please try again."))
        return {"status": "error", "error": str(e)}


@inngest_client.create_function(
    fn_id="RAG: Query with Image",
    trigger=inngest.TriggerEvent(event="rag/query_image_ai")
)
async def rag_query_image_ai(ctx: inngest.Context):
    user_id = ctx.event.data["user_id"]
    file_id = ctx.event.data["file_id"]
    caption = ctx.event.data.get("caption", "")
    language = ctx.event.data.get("language", "en")
    
    try:
        temp_path = await ctx.step.run("download-file", lambda: _download_telegram_file(file_id))
        extracted_text = await ctx.step.run(
            "extract-text-from-image",
            lambda: " ".join(load_and_chunk_image(temp_path))
        )

        try:
            os.remove(temp_path)  # Clean up the temporary file
        except FileNotFoundError:
            logging.warning(f"Temporary file already removed: {temp_path}")
        except Exception as cleanup_error:
            logging.warning(f"Failed to remove temp file {temp_path}: {cleanup_error}")

        if not extracted_text.strip():
            await ctx.step.run("send-no-text-reply", 
                lambda: _send_telegram_message(user_id, 
                    "I couldn't extract any text from the image. Please make sure the image contains clear, readable text."))
            return {"status": "no_text_extracted"}

        question = f"{caption}\n\n[Text extracted from image]:\n{extracted_text}".strip()
        
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, 5, language=language)

        if not found["contexts"]:
            await ctx.step.run("send-no-context-reply", 
                lambda: _send_telegram_message(user_id, 
                    "I couldn't find relevant information to answer your question. Please make sure educational content has been uploaded to the database. 📚"))
            return {"status": "no_context_found"}

        context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
        prompt = f"""
System Prompt: You are "Sahayak," a compassionate and expert AI teaching assistant for schools in rural India. Your primary goal is to make learning intuitive and relatable. Your tone should be encouraging, simple, and clear.
Your task is to answer the user's question, which is based on an image they sent. Follow these two steps precisely:
1.  **The Explanation:** First, provide a simple and factually correct explanation that directly answers the user's question. This explanation MUST be based strictly on the information within the "Provided Context" below. Do not add any information that is not in the context for this part.
2.  **The Relatable Example:** Second, and most importantly, create a vivid and simple analogy or example that connects the main point of the explanation to the daily life of a student in a rural Indian village. Use familiar elements like farming (kheti), cooking (rasoi), local nature (prakriti), a village market (bazaar), or common household items.
The entire response must be in the {language} language.
---
Provided Context:\n{context_block}\n---
User Question:\n{question}\n---
Answer:
"""
        answer = await ctx.step.run("llm-answer-image", lambda: _call_llm(prompt))
        await ctx.step.run("send-reply-image", lambda: _send_telegram_message(user_id, answer))
        return {"status": "success", "answer_sent": True}
    
    except Exception as e:
        logging.error(f"Error in rag_query_image_ai: {e}")
        await ctx.step.run("send-error-reply", 
            lambda: _send_telegram_message(user_id, 
                "Sorry, I encountered an error while processing your image. Please try again. 🙏"))
        return {"status": "error", "error": str(e)}

@inngest_client.create_function(
    fn_id="RAG: Handle Audio Query",
    trigger=inngest.TriggerEvent(event="rag/handle_audio_query")
)
async def rag_handle_audio_query(ctx: inngest.Context):
    user_id = ctx.event.data["user_id"]
    file_id = ctx.event.data["file_id"]
    
    temp_path = ""
    try:
        temp_path = await ctx.step.run("download-audio", 
            lambda: _download_telegram_file(file_id))
        
        question = await ctx.step.run("transcribe-audio", 
            lambda: _transcribe_audio(temp_path))
        
        if not question.strip():
            await ctx.step.run("send-no-text-reply", 
                lambda: _send_telegram_message(user_id, "I couldn't understand any speech in the audio. Please try again."))
            return {"status": "no_text_transcribed"}
        
        detected_lang = await ctx.step.run("detect-language",
            lambda: _detect_language(question))

        logging.info(f"Transcribed audio from {user_id}: '{question}' (lang: {detected_lang})")
        
        await ctx.step.send_event("send-rag-query", inngest.Event(
            name="rag/query_ai",
            data={
                "question": question,
                "user_id": user_id,
                "language": detected_lang
            }
        ))
        
        return {"status": "success", "forwarded_to_rag": True}
    
    except Exception as e:
        logging.error(f"Error in rag_handle_audio_query: {e}")
        await ctx.step.run("send-error-reply", 
            lambda: _send_telegram_message(user_id, 
                "Sorry, I. encountered an error processing your voice message. Please try again."))
        return {"status": "error", "error": str(e)}
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logging.warning(f"Failed to remove temp audio file {temp_path}: {cleanup_error}")

# --- FASTAPI APP & WEBHOOK ROUTER ---

app = FastAPI()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "app": "Sahayak RAG Bot",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/api/telegram/webhook",
            "inngest": "/api/inngest"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {}
    
    try:
        # Check Qdrant connection
        store = QdrantStorage()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    try:
        # Check Ollama connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    try:
        # Check Inngest connectivity
        inngest_status = "configured"
    except Exception as e:
        inngest_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if qdrant_status == "connected" and ollama_status == "connected" else "degraded",
        "services": {
            "qdrant": qdrant_status,
            "ollama": ollama_status,
            "inngest": inngest_status,
            "telegram": "configured" if TELEGRAM_BOT_TOKEN else "not configured"
        }
    }

@app.post("/api/telegram/webhook")
async def handle_telegram_webhook(request: dict):
    message = request.get("message")
    if not message:
        return {"status": "ok", "message": "Not a message update."}

    chat_id = str(message.get("chat", {}).get("id"))
    if not chat_id:
        return {"status": "ok", "message": "No chat ID."}
    
    try:
        if "text" in message:
            text_content = message["text"]
            detected_lang = _detect_language(text_content)
            
            logging.info(f"Received text message from {chat_id}, detected language: {detected_lang}")
            
            await inngest_client.send(inngest.Event(
                name="rag/query_ai", 
                data={
                    "question": text_content, 
                    "user_id": chat_id,
                    "language": detected_lang
                }
            ))
        
        elif "photo" in message:
            caption = message.get("caption", "")
            detected_lang = _detect_language(caption) if caption else DEFAULT_LANGUAGE
            
            logging.info(f"Received photo from {chat_id}, detected language: {detected_lang}")
            
            await inngest_client.send(inngest.Event(
                name="rag/query_image_ai", 
                data={
                    "file_id": message["photo"][-1]["file_id"], 
                    "caption": caption, 
                    "user_id": chat_id,
                    "language": detected_lang
                }
            ))

        elif "voice" in message:
            logging.info(f"Received voice message from {chat_id}")
            
            await inngest_client.send(inngest.Event(
                name="rag/handle_audio_query", 
                data={
                    "file_id": message["voice"]["file_id"], 
                    "user_id": chat_id
                }
            ))
        
        else:
            # Unsupported message type
            await _send_telegram_message(chat_id, "Sorry, I can only process text messages and images at the moment.")
            
    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        try:
            await _send_telegram_message(chat_id, "Sorry, I encountered an error processing your message. Please try again.")
        except:
            pass
        return {"status": "error", "message": str(e)}

    return {"status": "ok", "message": "Job dispatched."}

inngest.fast_api.serve(
    app, 
    inngest_client, 
    [
        rag_ingest_pdf, 
        rag_ingest_image, 
        rag_query_ai, 
        rag_query_image_ai,
        rag_handle_audio_query
    ]
)