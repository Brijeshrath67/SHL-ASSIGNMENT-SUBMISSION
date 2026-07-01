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
