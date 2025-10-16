import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime
import requests
import json
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc, language: str) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i], "language": language} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    language = ctx.event.data.get("language", "en")
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run(
        "embed-and-upsert", 
        lambda: _upsert(chunks_and_src, language),
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

    def _call_ollama(prompt_content: str) -> str:
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3:8b",
            "prompt": prompt_content,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        return json.loads(response.text)["response"].strip()

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    language = ctx.event.data.get("language", "en")

    found = await ctx.step.run(
        "embed-and-search", 
        lambda: _search(question, top_k, language),
        output_type=RAGSearchResult
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    
    prompt = f"""
System Prompt:
You are "Sahayak," a compassionate and expert AI teaching assistant for schools in rural India. Your primary goal is to make learning intuitive and relatable.
Follow these two steps in your answer:
1.  **The Explanation:** First, provide a simple and factually correct explanation for the user's question. This explanation MUST be based strictly on the provided context.
2.  **The Relatable Example:** Second, create a vivid and simple analogy or example that a student from a rural Indian village can deeply connect with. Use elements from their daily life, such as farming (kheti), cooking (rasoi), or local nature (prakriti).

The entire response must be in the {language} language.
---
Provided Context:
{context_block}
---
User Question:
{question}
---
Answer:
"""
    answer = await ctx.step.run("llm-answer", lambda: _call_ollama(prompt))
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])