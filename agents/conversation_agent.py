# agents/conversation_agent.py
import asyncio
import logging
from typing import Optional
from pipecat.frames.frames import AudioRawFrame, TextFrame, SystemFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from livekit import agents

from .config import AgentConfig
from processors.stt import ElevenLabsScribeSTT
from processors.interruption import InterruptionDetector
from .mcp_client import MCPToolClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_context = None
        self.pipeline = None
        self.runner = None
        self.mcp_client: Optional[MCPToolClient] = None

    async def setup(self):
        """Initialize all components"""
        logger.info("Setting up conversational agent...")

        # 1. STT (ElevenLabs Scribe v2 - 150ms latency)
        stt = ElevenLabsScribeSTT(
            api_key=self.config.elevenlabs_api_key,
            chunk_size=self.config.stt_chunk_size,
            model="scribe_v2_turbo"
        )

        # 2. MCP client - connects to the travel-tools MCP server (stdio) and
        # discovers its tools at runtime, instead of hardcoding schemas here.
        # Swap MCPToolClient's command/args to point at a different MCP
        # server (e.g. an official Google Maps MCP server) without touching
        # anything below this block.
        self.mcp_client = MCPToolClient()
        await self.mcp_client.connect()
        mcp_tools = await self.mcp_client.list_openai_tools()

        # 3. LLM Context (GPT-4.1 with streaming) + MCP-discovered tools.
        # LLMContext (new API) no longer takes model/system_prompt directly:
        # the system prompt is just the first message, and tools must be
        # FunctionSchema objects wrapped in a ToolsSchema, not raw
        # OpenAI-style dicts - so convert what the MCP client handed us.
        tool_schemas = [
            FunctionSchema(
                name=tool["function"]["name"],
                description=tool["function"].get("description", ""),
                properties=tool["function"].get("parameters", {}).get("properties", {}),
                required=tool["function"].get("parameters", {}).get("required", []),
            )
            for tool in mcp_tools
        ]
        self.llm_context = LLMContext(
            messages=[{"role": "system", "content": self.config.system_prompt}],
            tools=ToolsSchema(standard_tools=tool_schemas) if tool_schemas else None,
        )
        # User/assistant aggregators are the actual FrameProcessors that go
        # into the pipeline - LLMContext itself is just a data holder now,
        # it can't be placed in the Pipeline list directly.
        context_aggregator = LLMContextAggregatorPair(self.llm_context)

        # 4. LLM Service (streaming for faster TTFT)
        llm = OpenAILLMService(
            api_key=self.config.openai_api_key,
            model="gpt-4.1",
            max_tokens=self.config.llm_max_tokens,
            temperature=0.7,  # Conversational, not deterministic
            frequency_penalty=0.1,  # Less repetition
            presence_penalty=0.1,  # More variety
        )

        # 4b. Register each MCP tool with the LLM - calls get proxied over
        # stdio to the MCP server rather than running in-process.
        for tool in mcp_tools:
            name = tool["function"]["name"]
            llm.register_function(name, self.mcp_client.make_handler(name))

        # 5. TTS (ElevenLabs Flash v2.5 - WebSocket streaming).
        # chunk_length_schedule/optimize_streaming_latency no longer exist on
        # this service (ElevenLabs deprecated them for the newer streaming
        # protocol pipecat now targets) - dropped rather than guessed at.
        tts = ElevenLabsTTSService(
            api_key=self.config.elevenlabs_api_key,
            model=self.config.elevenlabs_model,
            voice_id=self.config.elevenlabs_voice_id,
        )

        # 6. Interruption detector (for turn-taking)
        interruption = InterruptionDetector(
            silence_threshold_ms=500,  # User stopped speaking
            interrupt_threshold_ms=300,  # User speaks during agent
        )

        # 7. LiveKit Transport (WebRTC).
        # participant identity now comes from the token itself (already set
        # via identity= in _generate_livekit_token), not a constructor arg.
        # audio_in_enabled/audio_out_enabled default to False in this
        # version - without them no audio flows at all.
        transport = LiveKitTransport(
            url=self.config.livekit_url,
            token=self._generate_livekit_token(),
            room_name="voice-agent-room",
            params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
        )

        # 8. Build pipeline - context_aggregator.user() captures the user's
        # turn into the context before the LLM call; .assistant() captures
        # the LLM's reply into the context after it's spoken.
        self.pipeline = Pipeline(
            [
                transport.input(),           # User audio in
                stt,                          # Speech → Text
                interruption,                 # Detect interrupts
                context_aggregator.user(),    # Text → LLM context
                llm,                          # LLM inference (streaming)
                tts,                          # Text → Audio (streaming)
                transport.output(),           # Audio out
                context_aggregator.assistant(),  # Record LLM reply into context
            ]
        )

        # 9. Runner
        self.runner = PipelineRunner()
        logger.info("Agent setup complete")

    async def start(self):
        """Start the agent"""
        await self.setup()
        try:
            logger.info(f"Starting {self.config.agent_name}...")
            await self.runner.run(self.pipeline)
        finally:
            await self.stop()

    async def stop(self):
        """Tear down external connections (MCP subprocess, etc.)"""
        if self.mcp_client:
            await self.mcp_client.close()
            self.mcp_client = None

    def _generate_livekit_token(self) -> str:
        """Generate LiveKit access token"""
        token = agents.TokenGenerator(
            api_key=self.config.livekit_api_key,
            api_secret=self.config.livekit_api_secret,
        ).generate(
            room="voice-agent-room",
            identity=self.config.agent_name,
            grants={
                "canPublish": True,
                "canPublishData": True,
                "canSubscribe": True,
            }
        )
        return token