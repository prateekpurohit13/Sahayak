# Sahayak-AI: RAG-based Teaching Assistant

This project implements a RAG (Retrieval-Augmented Generation) pipeline using FastAPI, Inngest, Qdrant, and Ollama to create an AI teaching assistant.

## Prerequisites

- [Python 3.13+](https://www.python.org/)
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Node.js and npm](https://nodejs.org/) (for the Inngest Dev Server)
- `uv` (Python package manager) installed (`pip install uv`)
- `ffmpeg` Required for audio processing (for the voice-to-text feature)

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Initialize Environment and Install Dependencies

First, set up the virtual environment and install the required Python packages using `uv`.

```bash
# Initialize a new virtual environment
uv venv

# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

### 2. Start Qdrant Vector Database

Run a Qdrant instance using Docker. This command mounts a local directory (`./qdrant_storage`) to persist the vector data.

```bash
docker run -d --name qdrantRAG -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 3. Run the FastAPI Application

Start the main application server with `uvicorn`. Using the `--reload` flag will automatically restart the server when you make code changes.

```bash
uv run uvicorn main:app --reload
```

The application will be running at `http://127.0.0.1:8000`.

### 4. Start the Inngest Dev Server

In a separate terminal, run the Inngest Dev Server. This tool provides a UI to inspect events and manually trigger your functions.

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

### 5. Interact with Your RAG Functions

Now that everything is running, you can use the Inngest Dev Server UI to interact with your functions.

1.  Open your browser and navigate to the Inngest Dev Server URL (usually `http://localhost:8288`).
2.  Click on the **Functions** tab in the sidebar. You will see `RAG: Ingest PDF` and `RAG: Query PDF`.

#### To Ingest a PDF:

1.  Find the **`RAG: Ingest PDF`** function and click the **Invoke** button.
2.  In the **Data** field, provide the path to the PDF you want to ingest. Make sure the PDF is in the `data` directory.

    ```json
    {
      "pdf_path": "./data/gecu104.pdf",
      "language": "en" //en for english and hi for hindi
    }
    ```

3.  Click **Invoke** to trigger the function.

#### To Ask a Question:

1.  Find the **`RAG: Query PDF`** function and click the **Invoke** button.
2.  In the **Data** field, enter your question.

    ```json
    {
      "question": "What is the main topic of the document?",
      "language": "en"
    }
    ```

3.  Click **Invoke** to get an answer based on the ingested content.

### 6. Interact from Telegram
#### setup instruction
1. Download ngrok, configure it and fetch the https://username.ngrok.io/
2. Run this  curl -Method POST "https://api.telegram.org/bot<TELEGRAM_API_TOKEN>/setWebhook?url=https://username.ngrok.io/api/telegram/webhook"

### 7. How to Use the Bot
Your setup is now complete. Open your Telegram bot and start a conversation.

**Asking a Question (Text)**
Simply type your question and send it. The bot will receive the text, generate an answer from the ingested documents, and reply.

**Asking a Question (Voice)**
1. Tap the microphone icon in your Telegram chat.
2. Record your question as a voice note.
3. Send the voice note.
4. The backend will automatically transcribe the audio to text, use that text to query the RAG pipeline, and send back the answer.