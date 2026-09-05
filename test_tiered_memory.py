"""Tests for tiny-agent-memory-v2."""
import pytest
from tiny_agent_memory import (
    TieredMemory, WorkingMemory, EpisodicMemory, SemanticMemory,
    Episode, SemanticFact
)


class TestWorkingMemory:
    def test_set_get(self):
        w = WorkingMemory()
        w["key"] = "value"
        assert w["key"] == "value"

    def test_get_missing(self):
        w = WorkingMemory()
        assert w.get("missing") is None

    def test_stats(self):
        w = WorkingMemory()
        w["x"] = 1
        _ = w["x"]  # hit
        _ = w.get("y")  # miss
        stats = w.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_expiry(self):
        w = WorkingMemory(max_age_seconds=0.1)
        w["x"] = 1
        import time; time.sleep(0.15)
        assert w.get("x") is None

    def test_max_size_eviction(self):
        w = WorkingMemory(max_size=2)
        w["a"] = 1
        w["b"] = 2
        w["c"] = 3
        assert w.get("a") is None
        assert w["b"] == 2
        assert w["c"] == 3

    def test_clear(self):
        w = WorkingMemory()
        w["x"] = 1
        w.clear()
        assert len(w) == 0
        stats = w.stats()
        assert stats["hits"] == 0

    def test_snapshot(self):
        w = WorkingMemory()
        w["x"] = 1
        w["y"] = 2
        snap = w.snapshot()
        assert snap == {"x": 1, "y": 2}


class TestEpisodicMemory:
    def test_remember(self):
        e = EpisodicMemory(agent_id="test", db_path=":memory:")
        ep = e.remember("Did the thing", outcome="success", tags=["test"])
        assert ep.content == "Did the thing"
        assert ep.outcome == "success"
        assert "test" in ep.tags

    def test_recall(self):
        e = EpisodicMemory(agent_id="test", db_path=":memory:")
        e.remember("Step 1")
        e.remember("Step 2")
        eps = e.recall(limit=10)
        assert len(eps) == 2
        assert eps[0].content == "Step 2"  # most recent first

    def test_search(self):
        e = EpisodicMemory(agent_id="test", db_path=":memory:")
        e.remember("Fixed the auth bug", tags=["auth", "fix"])
        e.remember("Fixed the DB bug", tags=["db", "fix"])
        results = e.search("auth", limit=5)
        assert len(results) == 1
        assert "auth" in results[0].content


class TestSemanticMemory:
    def test_store_recall(self):
        s = SemanticMemory(agent_id="test")
        s.store("The API uses JWT tokens")
        s.store("The DB is PostgreSQL")
        results = s.recall("jwt", limit=5)
        assert len(results) >= 1
        assert any("JWT" in f.fact for f in results)

    def test_use_count_bump(self):
        s = SemanticMemory(agent_id="test")
        s.store("Python 3.11 is faster")
        r1 = s.recall("python", limit=1)
        import time; time.sleep(0.01)
        r2 = s.recall("python", limit=1)
        assert r2[0].use_count >= r1[0].use_count


class TestTieredMemory:
    def test_full_pipeline(self):
        m = TieredMemory(agent_id="test-agent", db_path=":memory:")
        m.working["task"] = "Build the thing"
        m.working["error"] = "KeyError"
        ep = m.remember("Reviewed the code, found the issue", outcome="failure", tags=["review"])
        fact = m.store("The auth module uses Bearer tokens")
        results = m.recall("auth token")
        assert "episodes" in results
        assert "facts" in results
        assert len(results["facts"]) >= 1

    def test_snapshot(self):
        m = TieredMemory(agent_id="test", db_path=":memory:")
        m.working["x"] = 1
        m.remember("Did something")
        m.store("A fact")
        snap = m.snapshot()
        assert snap.agent_id == "test"
        assert snap.working_keys == 1
        assert snap.episode_count == 1
        assert snap.fact_count >= 1

    def test_stats(self):
        m = TieredMemory(agent_id="test", db_path=":memory:")
        s = m.stats()
        assert "working" in s
        assert "episodic" in s
        assert "semantic" in s
        assert s["agent_id"] == "test"

    def test_export(self):
        m = TieredMemory(agent_id="test", db_path=":memory:")
        m.remember("Test episode")
        m.store("Test fact")
        exp = m.export()
        assert exp["agent_id"] == "test"
        assert "exported_at" in exp
        assert "snapshot" in exp
