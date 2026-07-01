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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    display: flex; justify-content: center; align-items: center; min-height: 100vh;
    padding: 20px;
  }
  .chat {
    width: 640px; max-width: 100%; height: 85vh;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .header {
    padding: 20px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex; align-items: center; gap: 14px;
  }
  .header-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: white;
    flex-shrink: 0;
  }
  .header-text h1 { font-size: 17px; font-weight: 600; color: #fff; letter-spacing: -0.3px; }
  .header-text p { font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 2px; }
  .messages {
    flex: 1; overflow-y: auto; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 12px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.15) transparent;
  }
  .messages::-webkit-scrollbar { width: 4px; }
  .messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
  .msg {
    max-width: 85%; padding: 12px 16px;
    border-radius: 16px; font-size: 14px; line-height: 1.6;
    white-space: pre-wrap; animation: fadeIn 0.25s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .msg.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
  }
  .msg.assistant {
    align-self: flex-start;
    background: rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.9);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.06);
  }
  .msg.assistant a { color: #a78bfa; text-decoration: none; }
  .msg.assistant a:hover { text-decoration: underline; }
  .rec-card {
    margin-top: 8px; padding: 10px 12px;
    background: rgba(255,255,255,0.06);
    border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);
  }
  .rec-card + .rec-card { margin-top: 6px; }
  .rec-card a { font-weight: 500; font-size: 13px; }
  .rec-tag {
    display: inline-block;
    font-size: 10px; padding: 2px 8px; border-radius: 6px;
    background: rgba(167,139,250,0.2); color: #a78bfa;
    margin-right: 6px; font-weight: 500;
  }
  .input-area {
    padding: 16px 24px 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    display: flex; gap: 10px;
  }
  .input-area input {
    flex: 1; padding: 12px 16px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; font-size: 14px; outline: none;
    color: white; font-family: inherit;
    transition: border-color 0.2s;
  }
  .input-area input::placeholder { color: rgba(255,255,255,0.3); }
  .input-area input:focus { border-color: rgba(167,139,250,0.5); }
  .input-area button {
    padding: 12px 22px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 14px;
    font-size: 14px; font-weight: 500; cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    font-family: inherit;
  }
  .input-area button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
  .input-area button:active { transform: translateY(0); }
  .input-area button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
  .typing {
    align-self: flex-start;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.4);
    padding: 12px 16px; border-radius: 16px; border-bottom-left-radius: 4px;
    font-size: 14px; display: flex; align-items: center; gap: 8px;
  }
  .typing-dot { width: 6px; height: 6px; border-radius: 50%; background: rgba(255,255,255,0.3); animation: bounce 1.2s infinite; }
  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-6px); } }
  .eoc-badge {
    align-self: center;
    font-size: 11px; color: rgba(255,255,255,0.3);
    padding: 4px 12px; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; margin-top: 4px;
  }
</style>
</head>
<body>
<div class="chat">
  <div class="header">
    <div class="header-icon">&#9670;</div>
    <div class="header-text">
      <h1>SHL Assessment Advisor</h1>
      <p>AI-powered hiring assessment recommendations</p>
    </div>
  </div>
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <input id="input" type="text" placeholder="Describe the role you're hiring for..." autofocus>
    <button id="send" onclick="send()">Send</button>
  </div>
</div>
<script>
const messages = [];
let loading = false;

function addMsg(role, text, recs) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  if (role === 'user') {
    el.textContent = text;
  } else if (recs && recs.length) {
    let html = text.replace(/\\n/g, '<br>');
    html += '<div style="margin-top:10px;">';
    recs.forEach(r => {
      html += '<div class="rec-card"><span class="rec-tag">' + r.test_type.join(', ') + '</span><a href="' + r.url + '" target="_blank">' + r.name + '</a></div>';
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
  el.innerHTML = '<span>Thinking</span><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
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
      const eoc = document.createElement('div');
      eoc.className = 'eoc-badge';
      eoc.textContent = 'Conversation ended';
      document.getElementById('messages').appendChild(eoc);
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
