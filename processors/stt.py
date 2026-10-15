# processors/stt.py
import asyncio
import logging

import aiohttp
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class ElevenLabsScribeSTT(FrameProcessor):
    """
    ElevenLabs Scribe v2 Realtime STT processor.
    ~150ms latency, streaming transcription.

    Plain FrameProcessor has no separate start()/stop() lifecycle hooks in
    this pipecat version - StartFrame/EndFrame/CancelFrame arrive through
    process_frame() like any other frame, so connection setup/teardown
    happens there instead of in overridden methods.
    """

    def __init__(self, api_key: str, chunk_size: int = 512, model: str = "scribe_v2_turbo"):
        super().__init__()
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.model = model
        self.session: aiohttp.ClientSession | None = None
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.buffer = bytearray()

    async def _connect(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.session = aiohttp.ClientSession()
        self.websocket = await self.session.ws_connect(
            "wss://api.elevenlabs.io/v1/speech-to-text",
            headers=headers,
        )
        await self.websocket.send_json(
            {
                "model_id": self.model,
                "language": "en",
                "encoding": "pcm_16000",
            }
        )
        logger.info("ElevenLabs STT WebSocket connected")

    async def _disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        if self.session:
            await self.session.close()
            self.session = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._connect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame):
            if not self.websocket:
                # Socket not ready yet (audio arrived before StartFrame was
                # handled) - drop it rather than forwarding raw audio
                # downstream, where a text-expecting processor would choke.
                return

            self.buffer.extend(frame.audio)
            if len(self.buffer) >= self.chunk_size:
                chunk = bytes(self.buffer[: self.chunk_size])
                self.buffer = self.buffer[self.chunk_size :]

                await self.websocket.send_bytes(chunk)
                try:
                    async with asyncio.timeout(0.05):
                        msg = await self.websocket.receive_json()
                        if msg.get("type") == "transcript" and msg.get("text"):
                            await self.push_frame(TextFrame(text=msg["text"]))
                except asyncio.TimeoutError:
                    pass
            return

        # Any other frame type passes through unchanged.
        await self.push_frame(frame, direction)
