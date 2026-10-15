# processors/interruption.py
import logging

import numpy as np
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)


class InterruptionDetector(FrameProcessor):
    """
    Detects when the user speaks while the agent is speaking (barge-in).

    Tracks agent-speaking state from BotStartedSpeakingFrame /
    BotStoppedSpeakingFrame, which the TTS service emits as it plays audio -
    those frames flow back through the pipeline, so this processor needs to
    sit somewhere both directions reach it (or be linked into both the
    user-audio path and the bot-audio path) to see them.
    """

    def __init__(
        self,
        silence_threshold_ms: int = 500,
        interrupt_threshold_ms: int = 300,
        energy_threshold: float = 0.02,
    ):
        super().__init__()
        self.silence_threshold_ms = silence_threshold_ms
        self.interrupt_threshold_ms = interrupt_threshold_ms
        self.energy_threshold = energy_threshold
        self.agent_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self.agent_speaking = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self.agent_speaking = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame):
            audio_data = np.frombuffer(frame.audio, dtype=np.int16).astype(float)
            energy = np.sqrt(np.mean(audio_data**2))
            normalized_energy = energy / (2**15)
            is_speech = normalized_energy > self.energy_threshold

            if is_speech and self.agent_speaking:
                logger.info("[INTERRUPT] User barged in")
                await self.push_frame(InterruptionFrame())

            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
