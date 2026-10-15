# Aura Voice Agent

A real-time conversational voice agent: streaming speech-to-text → LLM
reasoning with tool-calling → streaming text-to-speech, over WebRTC via
LiveKit, with a multi-agent router on top for intent-specific tool scoping.

## Architecture

```
User audio (WebRTC/LiveKit)
   │
   ▼
STT (ElevenLabs Scribe, streaming)
   │
   ▼
Interruption detector (barge-in handling)
   │
   ▼
LangGraph router ── classifies intent ──► one of 5 specialist agents:
                                             travel / booking / support /
                                             smalltalk / escalation
                                             (each scoped to its own
                                             prompt + tools)
   │
   ▼
LLM (GPT-4.1, streaming, tool-calling)
   │
   ▼
TTS (ElevenLabs, streaming)
   │
   ▼
Agent audio out (WebRTC/LiveKit)
```

**Tool execution** happens two ways:
- Travel tools (`pick_random_place`, `plan_itinerary`) run behind an **MCP
  server** ([`mcp_server/travel_server.py`](mcp_server/travel_server.py)),
  called over stdio by an MCP client in the agent — this keeps tool
  implementation decoupled from the conversational agent, so a different
  MCP server (e.g. a hosted Maps provider's own server) can be swapped in
  by changing one client config, not the agent code.
- Booking/support tools are in-process mocks (in-memory data) — see
  [Known limitations](#known-limitations).

## Project layout

```
main.py                     FastAPI app + LiveKit worker entrypoint
agents/
  config.py                 Env-driven config (API keys, prompts, tuning)
  conversation_agent.py     Builds the pipecat pipeline (STT→LLM→TTS→transport)
  router.py                 LangGraph multi-agent router
  metrics.py                LLM-call counter (per node, per session)
  mcp_client.py              MCP client - talks to mcp_server/ over stdio
  tools.py                  Travel tool implementations (geocoding, itinerary)
  specialists/               5 specialist agents (prompt + tool subset each)
mcp_server/
  travel_server.py          MCP server wrapping agents/tools.py
processors/
  stt.py                     ElevenLabs STT as a pipecat FrameProcessor
  interruption.py            Barge-in detector as a pipecat FrameProcessor
eval/
  conversations.py          Fixed test conversation set
  baseline_agent.py          "Before" comparison: one agent, every tool
  run_baseline_vs_router.py  Measures LLM-call reduction from routing
```

## Setup

```bash
python -m venv voice_agent
source voice_agent/bin/activate    # or voice_agent\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env               # then fill in real values
```

Required environment variables (see `.env.example`):

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | LLM calls (GPT-4.1) |
| `ELEVENLABS_API_KEY` | STT + TTS |
| `LIVEKIT_URL` | LiveKit server address |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit room token generation |

**Never commit `.env`** — it's gitignored. If you're cloning this and your
copy lives inside a cloud-synced folder (OneDrive, Dropbox, etc.), be aware
that's a separate exposure path from git and isn't covered by `.gitignore`.

## Running

Local infra (LiveKit server + Redis) via Docker:
```bash
docker-compose up
```

The agent itself:
```bash
python -m main
```
(Must be run as a module — `agents/conversation_agent.py` uses relative
imports.)

## Measuring the multi-agent router's effect

The router's value proposition — fewer LLM calls than a single agent
seeing every tool every turn — is measured, not assumed:
```bash
python -m eval.run_baseline_vs_router
```
This replays `eval/conversations.py` through both systems and prints the
actual call-count reduction. Requires a real `OPENAI_API_KEY` and makes
live API calls (small cost).

## Known limitations

- **Live pipeline is unverified end-to-end.** The pipecat API surface this
  was originally written against had moved on (module paths, constructor
  args) by the time of the last audit in this repo's history; imports and
  object construction have been verified against the installed
  `pipecat-ai` version, but a full live call (real STT→LLM→TTS→WebRTC
  round trip) has not been run — doing that requires real API keys and a
  running LiveKit server, neither available in every dev environment.
- **The eval conversation set is small and hand-written** (not sampled
  from real usage).
- **No latency instrumentation yet** — the streaming architecture is
  designed around low time-to-first-audio, but no p50/p95 numbers are
  measured yet.
