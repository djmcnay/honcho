#!/usr/bin/env python3
"""
Honcho Pipeline Integration Tests

Verifies the Honcho embedding pipeline is actually working end-to-end,
not just that containers are up. Designed to be called from the health check.

Usage:
    docker exec -w /app honcho-api-1 /app/.venv/bin/python /app/tests/test_pipeline.py

Exit codes:
    0 = all tests passed
    1 = one or more tests failed

Output: TAP-style (Test Anything Protocol) for easy parsing.
"""
import asyncio
import json
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING)

# ─── Results ────────────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details: list[dict] = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        msg = f"ok {self.passed + self.failed} - {name}"
        if detail:
            msg += f"  # {detail}"
        print(msg)
        self.details.append({"name": name, "status": "pass", "detail": detail})

    def fail(self, name: str, detail: str = ""):
        self.failed += 1
        msg = f"not ok {self.passed + self.failed} - {name}"
        if detail:
            msg += f"  # {detail}"
        print(msg)
        self.details.append({"name": name, "status": "fail", "detail": detail})

    def skip(self, name: str, reason: str = ""):
        self.skipped += 1
        msg = f"ok {self.passed + self.failed + self.skipped} - {name}  # SKIP"
        if reason:
            msg += f" {reason}"
        print(msg)
        self.details.append({"name": name, "status": "skip", "detail": reason})

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.passed + self.failed + self.skipped,
            "details": self.details,
        }


# ─── Tests ──────────────────────────────────────────────────────────────────

async def test_config(r: Results):
    """Settings are correct for embedding pipeline."""
    try:
        from src.config import settings
    except Exception as e:
        r.fail("config_import", str(e))
        return

    if settings.EMBED_MESSAGES:
        r.ok("config_embed_messages", f"EMBED_MESSAGES={settings.EMBED_MESSAGES}")
    else:
        r.fail("config_embed_messages", f"EMBED_MESSAGES={settings.EMBED_MESSAGES} — embeddings disabled!")

    if settings.LLM.EMBEDDING_PROVIDER:
        r.ok("config_embedding_provider", f"provider={settings.LLM.EMBEDDING_PROVIDER}")
    else:
        r.fail("config_embedding_provider", "no provider configured")

    if settings.VECTOR_STORE.TYPE:
        r.ok("config_vector_store", f"type={settings.VECTOR_STORE.TYPE}")
    else:
        r.fail("config_vector_store", "no vector store type configured")


async def test_embedding_client(r: Results):
    """Embedding client can generate vectors."""
    try:
        from src.embedding_client import embedding_client
    except Exception as e:
        r.fail("embedding_client_import", str(e))
        return

    try:
        t0 = time.monotonic()
        vec = await embedding_client.embed("health check test query")
        elapsed = time.monotonic() - t0

        if vec and len(vec) == 1536:
            r.ok("embedding_client_embed", f"dim=1536, {elapsed:.2f}s")
        else:
            dim = len(vec) if vec else "None"
            r.fail("embedding_client_embed", f"unexpected dim={dim}")
    except Exception as e:
        r.fail("embedding_client_embed", f"{type(e).__name__}: {e}")


async def test_batch_embed(r: Results):
    """Batch embedding works for multiple texts."""
    try:
        from src.embedding_client import embedding_client
    except Exception as e:
        r.skip("batch_embed", f"import failed: {e}")
        return

    try:
        t0 = time.monotonic()
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        texts = {
            "test-1": ("hello world", enc.encode("hello world")),
            "test-2": ("honcho health check", enc.encode("honcho health check")),
        }
        result = await embedding_client.batch_embed(texts)
        elapsed = time.monotonic() - t0

        if len(result) == 2 and all(len(v) > 0 for v in result.values()):
            r.ok("batch_embed", f"2/2 embedded, {elapsed:.2f}s")
        else:
            r.fail("batch_embed", f"got {len(result)}/2 results")
    except Exception as e:
        r.fail("batch_embed", f"{type(e).__name__}: {e}")


async def test_db_counts(r: Results):
    """Message and embedding counts are reasonable."""
    try:
        from sqlalchemy import select, func
        from src import models
        from src.dependencies import tracked_db
    except Exception as e:
        r.skip("db_counts", f"import failed: {e}")
        return

    try:
        async with tracked_db("health_test") as db:
            msg_count = (await db.execute(select(func.count(models.Message.id)))).scalar()
            emb_count = (await db.execute(select(func.count(models.MessageEmbedding.id)))).scalar()

            if msg_count == 0:
                r.skip("db_message_count", "no messages yet — fresh install")
                return

            ratio = emb_count / msg_count if msg_count > 0 else 0
            if ratio >= 0.95:
                r.ok("db_embedding_coverage", f"{emb_count}/{msg_count} ({ratio:.0%})")
            elif ratio >= 0.50:
                r.fail("db_embedding_coverage", f"{emb_count}/{msg_count} ({ratio:.0%}) — degraded")
            else:
                r.fail("db_embedding_coverage", f"{emb_count}/{msg_count} ({ratio:.0%}) — broken")
    except Exception as e:
        r.fail("db_counts", f"{type(e).__name__}: {e}")


async def test_vector_search(r: Results):
    """Vector similarity search returns relevant results."""
    try:
        from sqlalchemy import text
        from src.dependencies import tracked_db
        from src.embedding_client import embedding_client
    except Exception as e:
        r.skip("vector_search", f"import failed: {e}")
        return

    try:
        async with tracked_db("health_test") as db:
            # Check if any embeddings exist
            count = (await db.execute(
                text("SELECT COUNT(*) FROM message_embeddings WHERE embedding IS NOT NULL")
            )).scalar()

            if count == 0:
                r.skip("vector_search", "no embeddings with vectors yet")
                return

            # Generate query embedding
            query_vec = await embedding_client.embed("general conversation")

            # Run vector search
            result = await db.execute(
                text("""
                    SELECT content, 1 - (embedding <=> (:emb)::vector) as sim
                    FROM message_embeddings
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> (:emb)::vector
                    LIMIT 1
                """),
                {"emb": str(query_vec)}
            )
            row = result.fetchone()

            if row and row.sim > 0:
                r.ok("vector_search", f"top result sim={row.sim:.4f}")
            else:
                r.fail("vector_search", "no results or zero similarity")
    except Exception as e:
        r.fail("vector_search", f"{type(e).__name__}: {e}")


async def test_deriver_observations(r: Results):
    """Deriver has created documents recently."""
    try:
        from sqlalchemy import select, func, text
        from src import models
        from src.dependencies import tracked_db
    except Exception as e:
        r.skip("deriver_observations", f"import failed: {e}")
        return

    try:
        async with tracked_db("health_test") as db:
            total = (await db.execute(
                select(func.count()).select_from(models.Document)
                .where(models.Document.deleted_at.is_(None))
            )).scalar()

            recent = (await db.execute(
                select(func.count()).select_from(models.Document)
                .where(models.Document.deleted_at.is_(None))
                .where(models.Document.created_at > func.now() - text("INTERVAL '24 hours'"))
            )).scalar()

            # Check whether there were any messages to derive from
            msgs_24h = (await db.execute(
                select(func.count()).select_from(models.Message)
                .where(models.Message.created_at > func.now() - text("INTERVAL '24 hours'"))
            )).scalar()

            if total == 0:
                r.fail("deriver_observations_total", "zero documents — deriver not producing")
            else:
                r.ok("deriver_observations_total", f"{total} documents")

            if recent > 0:
                r.ok("deriver_observations_24h", f"{recent} in last 24h")
            elif msgs_24h == 0:
                r.ok("deriver_observations_24h", "0 docs in 24h but 0 msgs — quiet day, not stalled")
            else:
                r.fail("deriver_observations_24h", f"0 docs in 24h despite {msgs_24h} msgs — deriver may be stalled")
    except Exception as e:
        r.fail("deriver_observations", f"{type(e).__name__}: {e}")


async def test_embedding_vector_quality(r: Results):
    """Sample embedding vectors are non-trivial (not all zeros, reasonable magnitude)."""
    try:
        from sqlalchemy import text
        from src.dependencies import tracked_db
        import math
    except Exception as e:
        r.skip("vector_quality", f"import failed: {e}")
        return

    try:
        async with tracked_db("health_test") as db:
            result = await db.execute(
                text("""
                    SELECT embedding::text as emb_text
                    FROM message_embeddings
                    WHERE embedding IS NOT NULL
                    LIMIT 3
                """)
            )
            rows = result.fetchall()

            if not rows:
                r.skip("vector_quality", "no embeddings to check")
                return

            bad = 0
            for row in rows:
                # Parse the vector string "[0.1, 0.2, ...]" 
                vec_str = row.emb_text.strip("[]")
                vec = [float(x) for x in vec_str.split(",")]
                magnitude = math.sqrt(sum(x * x for x in vec))

                if magnitude < 0.01:
                    bad += 1  # near-zero vector = suspicious
                elif magnitude > 100:
                    bad += 1  # huge magnitude = suspicious

            if bad == 0:
                r.ok("vector_quality", f"{len(rows)} samples, all non-trivial")
            else:
                r.fail("vector_quality", f"{bad}/{len(rows)} suspicious vectors")
    except Exception as e:
        r.fail("vector_quality", f"{type(e).__name__}: {e}")


# ─── Runner ─────────────────────────────────────────────────────────────────

async def run_all():
    r = Results()
    print("TAP version 13")
    print(f"1..8")  # total test count

    tests = [
        test_config,
        test_embedding_client,
        test_batch_embed,
        test_db_counts,
        test_vector_search,
        test_deriver_observations,
        test_embedding_vector_quality,
    ]

    for test_fn in tests:
        try:
            await test_fn(r)
        except Exception as e:
            r.fail(test_fn.__name__, f"unexpected error: {type(e).__name__}: {e}")

    print(f"\n# {r.passed} passed, {r.failed} failed, {r.skipped} skipped")

    # JSON output for health check to parse
    summary = r.summary()
    print(f"\n__PIPELINE_TEST_JSON__:{json.dumps(summary)}")

    return 0 if r.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))
