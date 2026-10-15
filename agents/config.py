import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

def env(name: str, default: str | None = None):
    """ Read an environment variable when AgentConfig() is created, not at import time."""
    return field(default_factory=lambda: os.getenv(name, default))

@dataclass
class AgentConfig:
    # API Keys (load from env)
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY")
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET")

    # Agent behavior
    agent_name: str = "Aura"
    agent_role: str = "Converstional AI Assistant"

    # Latency running
    stt_chunk_size: int = 512 # bytes per chunk
    tts_chunk_threshold: int = 20 # characters before generating audio
    llm_max_tokens: int = 150 # Short, conversational responses

    # ElevenLabs TTS config
    elevenlabs_model: str = "eleven_turbo_v2_5"  # Fastest model
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Example
    elevenlabs_streaming_latency: int = 2  # 0-4 (lower = lower latency)
    
    # System prompt
    system_prompt: str = """You are a helpful, conversational AI assistant. 
    Keep responses concise (<50 words) and natural. You sound like a real person.
    Use natural pauses and filler words ("um", "uh", "you know") sparingly for naturalness.
    Never explain you're an AI unless asked.
    
    Guidelines:
    - If unsure, ask clarifying questions
    - Be conversational, not robotic
    - Respond to emotional cues in the user's voice
    - Think out loud briefly if the question is complex

    Travel planning:
    - If the user wants a surprise destination or doesn't know where to go,
      call the pick_random_place tool, then talk about that place naturally.
    - If the user wants a plan/itinerary for a place (randomly picked or named
      by them), call the plan_itinerary tool and summarize it conversationally
      rather than reading it verbatim.
    """


