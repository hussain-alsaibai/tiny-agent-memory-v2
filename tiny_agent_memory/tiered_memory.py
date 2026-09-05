"""
tiny-agent-memory v2 — Tiered Episodic + Semantic + Working Memory for AI Agents
Zero-dependency. MIT. ~550 lines.

Three-tier memory architecture:
  Tier 1 — Working: in-process dict, zero-latency scratchpad
  Tier 2 — Episodic: snapdb-backed, compressed episode store with TTL
  Tier 3 — Semantic: TF-IDF indexed facts, retrieved by keyword + recency

Usage:
    from tiny_agent_memory import TieredMemory
    mem = TieredMemory(agent_id="devin-1")
    mem.working["task"] = "Fix auth bug in login.py"
    mem.remember("Reviewed login.py, found token not persisted in session")
    facts = mem.recall("login", limit=5)
"""

from __future__ import annotations

import json
import time
import hashlib
import re
import math
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from threading import Lock

try:
    import snapdb
    _HAS_SNAPDB = True
except Exception:
    _HAS_SNAPDB = False


# ─────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A single episodic memory — a structured record of a past event."""
    id: str
    agent_id: str
    content: str
    timestamp: float
    tags: list[str] = field(default_factory=list)
    outcome: Optional[str] = None  # "success", "failure", "partial"
    tokens_estimate: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(**d)


@dataclass
class SemanticFact:
    """A single semantic memory — a piece of persistent knowledge."""
    id: str
    agent_id: str
    fact: str
    source: str = "memory"
    timestamp: float = 0.0
    use_count: int = 0
    last_used: float = 0.0
    verified: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SemanticFact:
        return cls(**d)


@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of the full memory system."""
    agent_id: str
    timestamp: float
    working_keys: int
    episode_count: int
    fact_count: int
    total_tokens_estimate: int
    top_tags: list[str]


# ─────────────────────────────────────────────────────────────
# Tier 1: Working Memory (in-process, zero-latency)
# ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    In-process scratchpad. Zero dependencies, zero latency.
    Auto-evicts oldest entries when max_size is exceeded.
    Thread-safe.
    """

    def __init__(self, max_size: int = 500, max_age_seconds: float = 3600):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._max_age = max_age_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def __repr__(self) -> str:
        return f"<WorkingMemory keys={len(self._store)} hits={self._hits} misses={self._misses}>"

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._store:
                return False
            val, ts = self._store[key]
            if time.time() - ts > self._max_age:
                del self._store[key]
                return False
            return True

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                raise KeyError(key)
            val, ts = self._store[key]
            if time.time() - ts > self._max_age:
                del self._store[key]
                self._misses += 1
                raise KeyError(f"key '{key}' expired")
            self._store.move_to_end(key)
            self._hits += 1
            return val

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, time.time())

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._store[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        with self._lock:
            now = time.time()
            expired = [k for k, (_, ts) in self._store.items() if now - ts > self._max_age]
            for k in expired:
                del self._store[k]
            return list(self._store.keys())

    def values(self) -> list[Any]:
        with self._lock:
            return [v for v, _ in self._store.values()]

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return [(k, v) for k, (v, _) in self._store.items()]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def prune_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            now = time.time()
            before = len(self._store)
            self._store = OrderedDict(
                (k, v) for k, v in self._store.items()
                if now - v[1] <= self._max_age
            )
            return before - len(self._store)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "keys": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "max_size": self._max_size,
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, (v, _) in self._store.items()}


# ─────────────────────────────────────────────────────────────
# Tier 2: Episodic Memory (snapdb-backed, compressed episodes)
# ─────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Snapdb-backed episodic store. Compresses old episodes automatically.
    Thread-safe.
    """

    _NS = "episodes"

    def __init__(
        self,
        agent_id: str,
        db_path: str = "./.agent-memory.db",
        max_episodes: int = 1000,
        compress_after: int = 50,
    ):
        self.agent_id = agent_id
        self.max_episodes = max_episodes
        self.compress_after = compress_after
        self._db = None
        self._db_path = db_path
        self._lock = Lock()
        self._episode_count = 0
        self._use_snapdb = _HAS_SNAPDB

        if self._use_snapdb:
            try:
                self._db = snapdb.SnapDB(db_path, namespace=self._NS)
            except Exception:
                self._use_snapdb = False
                self._episodes: list[Episode] = []
        else:
            self._episodes: list[Episode] = []

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def remember(
        self,
        content: str,
        outcome: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Episode:
        """Record a new episode. Returns the created Episode."""
        ep = Episode(
            id=self._hash(content + str(time.time())),
            agent_id=self.agent_id,
            content=content,
            timestamp=time.time(),
            tags=tags or [],
            outcome=outcome,
            tokens_estimate=len(content) // 4,
            metadata=metadata or {},
        )
        with self._lock:
            if self._use_snapdb:
                self._db.put(ep.id, json.dumps(ep.to_dict()))
            else:
                self._episodes.append(ep)
            self._episode_count += 1
            self._maybe_compress()
        return ep

    def recall(self, limit: int = 10) -> list[Episode]:
        """Return the most recent episodes."""
        with self._lock:
            if self._use_snapdb:
                raw = self._db.get_range(limit=limit, reverse=True)
                return [Episode.from_dict(json.loads(r)) for _, r in raw]
            return list(reversed(self._episodes[-limit:]))

    def search(self, query: str, limit: int = 10) -> list[Episode]:
        """Simple substring + keyword search."""
        q = query.lower()
        results: list[tuple[float, Episode]] = []
        with self._lock:
            if self._use_snapdb:
                raw = self._db.get_range(limit=1000)
                episodes = [Episode.from_dict(json.loads(r)) for _, r in raw]
            else:
                episodes = list(self._episodes)
        for ep in episodes:
            if q in ep.content.lower() or any(q in t.lower() for t in ep.tags):
                score = 0.0
                if q in ep.content.lower():
                    score += 0.7
                if any(q in t.lower() for t in ep.tags):
                    score += 0.3
                results.append((score, ep))
        results.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [ep for _, ep in results[:limit]]

    def count(self) -> int:
        with self._lock:
            if self._use_snapdb:
                return self._db.len()
            return len(self._episodes)

    def _maybe_compress(self) -> None:
        if self._episode_count % self.compress_after == 0:
            if self._use_snapdb:
                count = self._db.len()
                if count > self.max_episodes:
                    raw = self._db.get_range(limit=count)
                    old_count = max(0, count - self.max_episodes)
                    for i, (_, v) in enumerate(raw):
                        if i < old_count:
                            ep = Episode.from_dict(json.loads(v))
                            # Compress: keep id + timestamp + summary only
                            compressed = {
                                "id": ep.id,
                                "agent_id": ep.agent_id,
                                "content": ep.content[:80] + "..." if len(ep.content) > 80 else ep.content,
                                "timestamp": ep.timestamp,
                                "tags": ep.tags,
                                "outcome": ep.outcome,
                                "tokens_estimate": ep.tokens_estimate,
                                "metadata": {"compressed": True},
                            }
                            self._db.put(ep.id, json.dumps(compressed))
            else:
                if len(self._episodes) > self.max_episodes:
                    keep = self._episodes[-self.max_episodes:]
                    self._episodes = keep

    def stats(self) -> dict:
        with self._lock:
            return {
                "episode_count": self.count(),
                "max_episodes": self.max_episodes,
                "compressed_count": sum(
                    1 for e in (self._episodes if not self._use_snapdb else [])
                    if e.metadata.get("compressed")
                ),
            }


# ─────────────────────────────────────────────────────────────
# Tier 3: Semantic Memory (TF-IDF indexed knowledge base)
# ─────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Keyword + recency ranked fact store. No embeddings = zero deps + fast.
    For production use with vectors, layer on top of snapdb + your embedder.
    """

    _STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "my",
        "your", "his", "her", "its", "our", "their", "what", "which", "who",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "not", "only", "same",
        "so", "than", "too", "very", "just", "as", "if", "then", "because",
        "while", "although", "though", "after", "before", "above", "below",
        "between", "into", "through", "during", "about", "also", "any", "about",
    })

    def __init__(self, agent_id: str, db_path: str = "./.agent-memory.db"):
        self.agent_id = agent_id
        self._facts: OrderedDict[str, SemanticFact] = OrderedDict()
        self._use_snapdb = _HAS_SNAPDB
        self._db = None
        self._db_path = db_path
        self._lock = Lock()

        if self._use_snapdb:
            try:
                self._db = snapdb.SnapDB(db_path, namespace="semantic")
            except Exception:
                self._use_snapdb = False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
        return [t for t in tokens if t not in SemanticMemory._STOPWORDS]

    @staticmethod
    def _score_fact(fact: SemanticFact, query_tokens: list[str]) -> float:
        content_tokens = set(SemanticMemory._tokenize(fact.fact))
        if not content_tokens or not query_tokens:
            return 0.0
        overlap = content_tokens & set(query_tokens)
        tf = len(overlap) / len(content_tokens)
        idf = 1.0  # simplified: no IDF in zero-dep version
        recency_bonus = 1.0 + min(fact.use_count * 0.05, 0.5)
        age_penalty = max(0.1, 1.0 - (time.time() - fact.timestamp) / (30 * 86400))
        return tf * idf * recency_bonus * age_penalty

    def store(self, fact: str, source: str = "memory", metadata: Optional[dict] = None) -> SemanticFact:
        """Store a new fact."""
        sf = SemanticFact(
            id=hashlib.sha256(fact.encode()).hexdigest()[:16],
            agent_id=self.agent_id,
            fact=fact,
            source=source,
            timestamp=time.time(),
            use_count=0,
            last_used=0.0,
            verified=False,
            metadata=metadata or {},
        )
        with self._lock:
            self._facts[sf.id] = sf
            if self._use_snapdb and self._db:
                self._db.put(sf.id, json.dumps(sf.to_dict()))
        return sf

    def recall(self, query: str, limit: int = 5) -> list[SemanticFact]:
        """Rank-ordered fact retrieval."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        with self._lock:
            facts = list(self._facts.values())
        scored = [(self._score_fact(f, query_tokens), f) for f in facts]
        scored.sort(key=lambda x: -x[0])
        results = [f for _, f in scored[:limit]]
        # Update use counts
        now = time.time()
        with self._lock:
            for f in results:
                f.use_count += 1
                f.last_used = now
        return results

    def count(self) -> int:
        with self._lock:
            return len(self._facts)

    def stats(self) -> dict:
        with self._lock:
            total_uses = sum(f.use_count for f in self._facts.values())
            return {
                "fact_count": len(self._facts),
                "total_uses": total_uses,
                "verified_count": sum(1 for f in self._facts.values() if f.verified),
            }


# ─────────────────────────────────────────────────────────────
# Tiered Memory — Unified Interface
# ─────────────────────────────────────────────────────────────

class TieredMemory:
    """
    Unified three-tier memory system for AI agents.

    Usage:
        mem = TieredMemory(agent_id="agent-1", db_path="./memory.db")

        # Working — zero latency scratchpad
        mem.working["current_file"] = "src/auth.py"
        mem.working["error"] = "KeyError: 'token'"

        # Episodic — what happened
        mem.remember("Reviewed auth.py, found token not persisted in session storage")
        mem.remember("Applied fix, tests pass", outcome="success", tags=["auth", "fix"])

        # Semantic — what I know
        mem.store("The auth module uses JWT tokens stored in request.headers")
        mem.store("The session store is Redis at localhost:6379", source="config")

        # Recall — unified query
        facts = mem.recall("auth token session")
        print(facts)
    """

    def __init__(
        self,
        agent_id: str = "default",
        db_path: str = "./.agent-memory.db",
        working_max_size: int = 500,
        working_max_age_seconds: float = 3600.0,
        max_episodes: int = 1000,
    ):
        self.agent_id = agent_id
        self.working = WorkingMemory(max_size=working_max_size, max_age_seconds=working_max_age_seconds)
        self.episodic = EpisodicMemory(
            agent_id=agent_id,
            db_path=db_path,
            max_episodes=max_episodes,
        )
        self.semantic = SemanticMemory(agent_id=agent_id, db_path=db_path)

    # ── Convenience aliases ─────────────────────────────────

    def remember(
        self,
        content: str,
        outcome: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Episode:
        """Alias for episodic.remember()"""
        return self.episodic.remember(
            content=content,
            outcome=outcome,
            tags=tags,
            metadata=metadata,
        )

    def store(self, fact: str, source: str = "memory", metadata: Optional[dict] = None) -> SemanticFact:
        """Alias for semantic.store()"""
        return self.semantic.store(fact=fact, source=source, metadata=metadata)

    def recall(self, query: str, limit: int = 5) -> dict:
        """
        Unified recall across all three tiers.
        Returns dict with 'episodes' and 'facts' keys.
        """
        return {
            "episodes": self.episodic.search(query, limit=limit),
            "facts": self.semantic.recall(query, limit=limit),
        }

    # ── Snapshot ─────────────────────────────────────────────

    def snapshot(self) -> MemorySnapshot:
        """Full system snapshot for monitoring."""
        return MemorySnapshot(
            agent_id=self.agent_id,
            timestamp=time.time(),
            working_keys=len(self.working),
            episode_count=self.episodic.count(),
            fact_count=self.semantic.count(),
            total_tokens_estimate=(
                len(self.working) * 50 +
                self.episodic.count() * 200 +
                self.semantic.count() * 100
            ),
            top_tags=self._top_tags(),
        )

    def _top_tags(self, limit: int = 5) -> list[str]:
        tag_counts: dict[str, int] = {}
        for ep in self.episodic.recall(limit=200):
            for tag in ep.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return sorted(tag_counts, key=tag_counts.get, reverse=True)[:limit]

    def stats(self) -> dict:
        """Full system statistics."""
        return {
            "agent_id": self.agent_id,
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "snapshot": asdict(self.snapshot()),
        }

    def export(self) -> dict:
        """Export all memory as a serializable dict."""
        snap = self.snapshot()
        return {
            "agent_id": self.agent_id,
            "exported_at": time.time(),
            "snapshot": asdict(snap),
            "episodes": [e.to_dict() for e in self.episodic.recall(limit=self.episodic.count())],
            "facts": [f.to_dict() for f in self.semantic.recall(query="", limit=self.semantic.count())],
        }


# ─────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────

def main():
    import argparse, sys

    parser = argparse.ArgumentParser(description="tiny-agent-memory — Tiered Agent Memory")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("stats", help="Show memory statistics")
    p.add_argument("--db", default="./.agent-memory.db")
    p.add_argument("--agent-id", default="cli")

    p = sub.add_parser("export", help="Export memory to JSON")
    p.add_argument("--db", default="./.agent-memory.db")
    p.add_argument("--agent-id", default="cli")
    p.add_argument("--output", default="memory-export.json")

    p = sub.add_parser("remember", help="Record an episode")
    p.add_argument("--db", default="./.agent-memory.db")
    p.add_argument("--agent-id", default="cli")
    p.add_argument("content", help="Episode content")
    p.add_argument("--outcome")
    p.add_argument("--tags", help="Comma-separated tags")

    p = sub.add_parser("recall", help="Recall from memory")
    p.add_argument("--db", default="./.agent-memory.db")
    p.add_argument("--agent-id", default="cli")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.cmd in ("stats", "export", "remember", "recall"):
        mem = TieredMemory(agent_id=args.agent_id, db_path=args.db)
    else:
        parser.print_help()
        return

    if args.cmd == "stats":
        import json
        print(json.dumps(mem.stats(), indent=2))

    elif args.cmd == "export":
        import json
        with open(args.output, "w") as f:
            json.dump(mem.export(), f, indent=2)
        print(f"Exported to {args.output}")

    elif args.cmd == "remember":
        ep = mem.remember(
            content=args.content,
            outcome=args.outcome,
            tags=args.tags.split(",") if args.tags else None,
        )
        print(f"Episode recorded: {ep.id}")

    elif args.cmd == "recall":
        if args.query:
            results = mem.recall(args.query, limit=args.limit)
            print(f"Episodes: {len(results['episodes'])}")
            for ep in results["episodes"]:
                print(f"  [{ep.id}] {ep.content[:80]}")
            print(f"Facts: {len(results['facts'])}")
            for f in results["facts"]:
                print(f"  {f.fact[:80]}")
        else:
            eps = mem.episodic.recall(limit=args.limit)
            print(f"Recent episodes: {len(eps)}")
            for ep in eps:
                print(f"  [{ep.id[:8]}] {ep.content[:80]}")


if __name__ == "__main__":
    main()
