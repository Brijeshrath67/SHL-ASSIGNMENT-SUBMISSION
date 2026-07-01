from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .catalog import load_catalog
from .retrieval import CatalogRetriever
from .agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Advisor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

catalog: list[dict] = []
retriever: CatalogRetriever | None = None
agent: Agent | None = None


@app.on_event("startup")
async def startup():
    global catalog, retriever, agent
    logger.info("Loading catalog...")
    catalog = load_catalog()
    logger.info("Loaded %d catalog items", len(catalog))
    retriever = CatalogRetriever(catalog)
    agent = Agent(retriever)
    logger.info("Agent ready")


from fastapi.responses import HTMLResponse

CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHL Assessment Advisor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
  .chat { width: 600px; max-width: 95vw; height: 90vh; background: white; border-radius: 12px; box-shadow: 0 2px 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
  .header { padding: 16px 20px; border-bottom: 1px solid #eee; }
  .header h1 { font-size: 16px; color: #333; }
  .header p { font-size: 12px; color: #888; margin-top: 2px; }
  .messages { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #007bff; color: white; border-bottom-right-radius: 4px; }
  .msg.assistant { align-self: flex-start; background: #f0f0f0; color: #333; border-bottom-left-radius: 4px; }
  .msg.rec { align-self: flex-start; background: #e8f4e8; color: #333; border-bottom-left-radius: 4px; font-size: 13px; }
  .msg.rec a { color: #007bff; text-decoration: none; }
  .msg.rec a:hover { text-decoration: underline; }
  .input-area { padding: 12px 20px; border-top: 1px solid #eee; display: flex; gap: 8px; }
  .input-area input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }
  .input-area input:focus { border-color: #007bff; }
  .input-area button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
  .input-area button:disabled { opacity: 0.6; cursor: not-allowed; }
  .typing { align-self: flex-start; background: #f0f0f0; color: #888; padding: 10px 14px; border-radius: 12px; border-bottom-left-radius: 4px; font-size: 14px; }
  .rec-badge { display: inline-block; background: #28a745; color: white; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-top: 4px; }
</style>
</head>
<body>
<div class="chat" id="chat">
  <div class="header">
    <h1>SHL Assessment Advisor</h1>
    <p>Ask me about hiring assessments</p>
  </div>
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <input id="input" type="text" placeholder="Type your message..." autofocus>
    <button id="send" onclick="send()">Send</button>
  </div>
</div>
<script>
const messages = [];
let loading = false;

function addMsg(role, text, recs) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  if (recs && recs.length) {
    let html = text.replace(/\\n/g, '<br>');
    html += '<div style="margin-top:8px;font-size:12px;color:#666;">';
    recs.forEach(r => {
      html += '<div><span class="rec-badge">' + r.test_type.join(', ') + '</span> <a href="' + r.url + '" target="_blank">' + r.name + '</a></div>';
    });
    html += '</div>';
    el.innerHTML = html;
  } else {
    el.textContent = text;
  }
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({ behavior: 'smooth' });
}

function addTyping() {
  const el = document.createElement('div');
  el.className = 'typing';
  el.id = 'typing';
  el.textContent = 'Thinking...';
  document.getElementById('messages').appendChild(el);
  el.scrollIntoView({ behavior: 'smooth' });
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function send() {
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text || loading) return;
  input.value = '';
  loading = true;
  document.getElementById('send').disabled = true;

  messages.push({ role: 'user', content: text });
  addMsg('user', text);
  addTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages.map(m => ({ role: m.role, content: m.content })) })
    });
    const data = await res.json();
    removeTyping();
    addMsg('assistant', data.reply, data.recommendations);
    messages.push({ role: 'assistant', content: data.reply });
    if (data.end_of_conversation) {
      document.getElementById('input').disabled = true;
      document.getElementById('send').disabled = true;
    }
  } catch (e) {
    removeTyping();
    addMsg('assistant', 'Error: ' + e.message);
  }
  loading = false;
  document.getElementById('send').disabled = false;
}

document.getElementById('input').addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>
"""


@app.get("/")
async def index():
    return HTMLResponse(CHAT_HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        return ChatResponse(
            reply="Hello! I'm your SHL assessment advisor. How can I help you with your hiring needs today?",
            recommendations=[],
            end_of_conversation=False,
        )
    if not agent:
        return ChatResponse(
            reply="Service is still initializing. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )
    if len(req.messages) > 16:
        return ChatResponse(
            reply="This conversation has reached its maximum length. Please start a new conversation.",
            recommendations=[],
            end_of_conversation=True,
        )
    try:
        result = await agent.process(req.messages)
        return result
    except Exception as exc:
        logger.exception("Error processing chat request")
        return ChatResponse(
            reply="I apologize, but I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=True,
        )
