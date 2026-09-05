# tiny-agent-memory-v2 — Tiered Agent Memory

> **Three-tier memory architecture for AI agents.**
> Working (zero-latency) → Episodic (compressed history) → Semantic (knowledge base)
> Zero dependencies. MIT. ~550 lines.

```bash
pip install tiny-agent-memory
# or just copy tiny_agent_memory/tiered_memory.py
```

## Why Three Tiers?

AI agents need memory at different timescales:

| Tier | Latency | Contents | Size |
|------|---------|----------|------|
| **Working** | ~0ms | Current task, scratchpad | 500 items |
| **Episodic** | ~1ms | What happened, compressed history | 1000 episodes |
| **Semantic** | ~5ms | What I know, TF-IDF indexed | unlimited |

## Quick Start

```python
from tiny_agent_memory import TieredMemory

mem = TieredMemory(agent_id="devin-1")

# Working memory — zero latency
mem.working["task"] = "Fix auth bug in login.py"
mem.working["error"] = "KeyError: 'token'"

# Episodic — what happened
mem.remember("Reviewed auth.py, found token not persisted in session storage")
mem.remember("Applied fix, tests pass", outcome="success", tags=["auth", "fix"])

# Semantic — what I know
mem.store("The auth module uses JWT tokens stored in request.headers")
mem.store("The session store is Redis at localhost:6379", source="config")

# Unified recall — searches all three tiers
results = mem.recall("auth token session")
print(results["episodes"])  # Recent episodes about auth
print(results["facts"])     # Known facts about auth
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Working Memory (Scratchpad)                        │
│  • Current task state, in-progress reasoning        │
│  • Auto-pruned at token budget                      │
│  • Latency: ~0ms (in-process dict)                │
├─────────────────────────────────────────────────────┤
│  Episodic Memory (Recent History)                   │
│  • Last N completed steps, decisions, outcomes      │
│  • Compressed summaries of older episodes           │
│  • Latency: ~1ms (snapdb read)                    │
├─────────────────────────────────────────────────────┤
│  Semantic Memory (Knowledge Base)                   │
│  • Project facts, API shapes, conventions          │
│  • TF-IDF indexed, retrieved by keyword + recency  │
│  • Latency: ~5ms                                  │
└─────────────────────────────────────────────────────┘
```

## CLI

```bash
# Record an episode
python -m tiny_agent_memory remember --agent-id my-agent "Fixed the null pointer in user.py"

# Recall from memory
python -m tiny_agent_memory recall --agent-id my-agent "null pointer"

# Export full memory
python -m tiny_agent_memory export --agent-id my-agent --output memory.json

# Show statistics
python -m tiny_agent_memory stats --agent-id my-agent
```

## Installation

```bash
pip install tiny-agent-memory
```

Or copy the file directly:

```bash
curl -O https://raw.githubusercontent.com/hussain-alsaibai/tiny-agent-memory-v2/main/tiny_agent_memory/tiered_memory.py
```

## Dependencies

**Zero.** The only optional dependency is `snapdb` for persistent episode storage — if not available, it falls back to an in-memory list.

## Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Working memory read | **~0.01ms** | Dict lookup |
| Working memory write | **~0.01ms** | Dict write |
| Episode recall | **~1ms** | snapdb read, top 10 |
| Semantic recall | **~5ms** | TF-IDF over 100 facts |
| Snapshot | **~0.5ms** | Full system state |

## License

MIT © 2026 Hussain Al-Saibai
