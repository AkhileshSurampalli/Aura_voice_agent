# main.py
import logging
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import agents
from livekit.plugins.openai import LLM
from .agents.conversation_agent import ConversationAgent
from .agents.config import AgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice AI Agent")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_config = AgentConfig()
conversation_agent = None

@app.on_event("startup")
async def startup():
    """Initialize agent on startup"""
    global conversation_agent
    conversation_agent = ConversationAgent(agent_config)
    logger.info("Agent initialized")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": agent_config.agent_name}

@agents.prewarm_plugin
async def prewarm_fn(proc):
    """Pre-warm LLM on startup"""
    logger.info("Warming up LLM...")
    llm = LLM.with_model("gpt-4.1")
    await llm.agenerate(
        "Say 'Ready to assist' in one sentence."
    )
    logger.info("LLM ready")

@agents.worker_entrypoint("voice-agent")
async def agent_worker(ctx: agents.JobContext):
    """Main agent worker for LiveKit calls"""
    logger.info(f"Starting agent for room: {ctx.room.name}")
    
    # Create agent instance for this call
    agent = ConversationAgent(agent_config)
    await agent.start()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for browser-based testing"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            # Echo back for testing
            await websocket.send_json({
                "type": "response",
                "text": f"Echo: {data.get('text', '')}"
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )