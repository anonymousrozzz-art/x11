import os
import json
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CLIENT CONFIGURATION & INITIALIZATION ---

# Groq Client 1 (Handles: Llama 3.3, Kimi K2, ALLaM 7B)
GROQ_CLIENT_1 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_1")
)

# Groq Client 2 (Handles: Qwen 3, Compound, GPT-OSS 120B)
GROQ_CLIENT_2 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_2")
)

# SambaNova Client (Handles: Llama 3.3 70B)
SAMBANOVA_CLIENT = AsyncOpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.getenv("SAMBANOVA_API_KEY")
)

# Google Gemini Client (Handles: Flash 2.0)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cloudflare Configuration (Handles: Llama 3.2)
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CF_HEADERS = {"Authorization": f"Bearer {CF_API_TOKEN}"}

# --- HELPER FUNCTIONS ---

def build_memory_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Manually constructs a conversation script from the message history.
    This is required for stateless APIs (Pollinations, ApiFreeLLM, Cloudflare REST)
    that do not accept a list of message objects.
    """
    prompt = "Conversation History:\n"
    for m in messages:
        role = "User" if m['role'] == 'user' else "AI"
        prompt += f"{role}: {m['content']}\n"
    prompt += "AI:"
    return prompt

# --- 2. GENERATOR FUNCTIONS (STREAMING LOGIC) ---

# === GROQ ENGINE 1 MODELS ===

async def generate_groq_llama(messages):
    """Stream Llama 3.3 70B Versatile"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Llama Error: {str(e)}"

async def generate_groq_kimi(messages):
    """Stream Moonshot AI Kimi K2 Instruct"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Kimi Error: {str(e)}"

async def generate_groq_allam(messages):
    """Stream ALLaM 2 7B (Saudi AI)"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="allam-2-7b",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq ALLaM Error: {str(e)}"

# === GROQ ENGINE 2 MODELS ===

async def generate_groq_qwen(messages):
    """Stream Qwen 3 32B"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Qwen Error: {str(e)}"

async def generate_groq_compound(messages):
    """Stream Groq Compound (Agentic System)"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="groq/compound",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq Compound Error: {str(e)}"

async def generate_groq_gptoss(messages):
    """Stream OpenAI GPT-OSS 120B"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Groq GPT-OSS Error: {str(e)}"

# === OTHER PROVIDERS ===

async def generate_samba(messages):
    """Stream SambaNova Llama 3.3 70B"""
    try:
        stream = await SAMBANOVA_CLIENT.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"SambaNova Error: {str(e)}"

async def generate_gemini(messages):
    """Stream Gemini (Auto-switching versions)"""
    # Convert standard messages format to Gemini's history format
    gemini_history = []
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})
    
    current_message = messages[-1]["content"]

    # Priority list: Experimental -> New Stable -> Old Stable
    models_to_try = ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    for model_id in models_to_try:
        try:
            model = genai.GenerativeModel(model_id)
            chat = model.start_chat(history=gemini_history)
            response = await chat.send_message_async(current_message, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
            return # Exit function on success
        except Exception:
            continue # Try next model in list
            
    yield "Gemini Error: All model versions failed. Please check API Key or Quota."

async def generate_pollinations(messages):
    """Stream Pollinations (GPT-4o Proxy) - Stateless"""
    full_text = build_memory_prompt(messages)
    encoded_prompt = requests.utils.quote(full_text)
    # Using the 'openai' model which maps to GPT-4o
    url = f"https://text.pollinations.ai/{encoded_prompt}?model=openai"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                else:
                    yield f"Pollinations Error: HTTP {response.status}"
    except Exception as e:
        yield f"Pollinations Connection Error: {str(e)}"

async def generate_cloudflare(messages):
    """Stream Cloudflare Workers AI (Llama 3.2) - Stateless"""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.2-3b-instruct"
    full_text = build_memory_prompt(messages)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=CF_HEADERS, 
                json={"prompt": full_text, "stream": True}
            ) as response:
                async for line in response.content:
                    line_text = line.decode("utf-8").strip()
                    if line_text.startswith("data: "):
                        try:
                            # Cloudflare streams data: {"response": "word"}
                            data = json.loads(line_text[6:])
                            if "response" in data:
                                yield data["response"]
                        except:
                            pass
    except Exception as e:
        yield f"Cloudflare Error: {str(e)}"

async def generate_free(messages):
    """Fetch ApiFreeLLM (Synchronous) - Stateless"""
    url = "https://apifreellm.com/api/chat"
    full_text = build_memory_prompt(messages)
    
    try:
        # Using sync requests with timeout, yielded as a single chunk
        # The frontend handles the "typing" animation for this big chunk
        resp = requests.post(url, json={"message": full_text}, timeout=60)
        if resp.ok:
            json_response = resp.json()
            # Check for common response keys
            text = json_response.get("response") or json_response.get("message") or str(json_response)
            yield text
        else:
            yield f"ApiFreeLLM Error: Status {resp.status_code}"
    except Exception as e:
        yield f"ApiFreeLLM Connection Error: {str(e)}"

# --- 3. ROUTE HANDLERS ---

@app.get("/")
async def get_ui():
    """Serves the main HTML interface"""
    with open("index.html", "r", encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.post("/chat/{provider}")
async def chat_endpoint(provider: str, request: Request):
    """Main routing endpoint for all AI models"""
    try:
        data = await request.json()
        messages = data.get("messages", [])

        # Groq Key 1 Routes
        if provider == "groq_llama":
            return StreamingResponse(generate_groq_llama(messages), media_type="text/plain")
        if provider == "groq_kimi":
            return StreamingResponse(generate_groq_kimi(messages), media_type="text/plain")
        if provider == "groq_allam":
            return StreamingResponse(generate_groq_allam(messages), media_type="text/plain")
        
        # Groq Key 2 Routes
        if provider == "groq_qwen":
            return StreamingResponse(generate_groq_qwen(messages), media_type="text/plain")
        if provider == "groq_compound":
            return StreamingResponse(generate_groq_compound(messages), media_type="text/plain")
        if provider == "groq_gptoss":
            return StreamingResponse(generate_groq_gptoss(messages), media_type="text/plain")
        
        # Other Provider Routes
        if provider == "samba":
            return StreamingResponse(generate_samba(messages), media_type="text/plain")
        if provider == "gemini":
            return StreamingResponse(generate_gemini(messages), media_type="text/plain")
        if provider == "poll":
            return StreamingResponse(generate_pollinations(messages), media_type="text/plain")
        if provider == "cf":
            return StreamingResponse(generate_cloudflare(messages), media_type="text/plain")
        if provider == "free":
            return StreamingResponse(generate_free(messages), media_type="text/plain")
        
        return "Invalid Provider ID"
        
    except Exception as e:
        return f"Server Error: {str(e)}"