"""
ledger/index.py - building the search index.

Run it with:   uv run python -m ledger.index

For every downloaded filing:
  HTML --(parse)--> clean text --(chunk)--> chunks --(embed)--> vectors
Then:
  - clean text is saved to data/text/   (needed to verify exact citations)
  - chunks are saved to data/chunks.jsonl (needed for keyword search + display)
  - vectors are stored in Qdrant          (needed for meaning-based search)

Re-run this whenever you change parsing, chunking, or the embedding model.
It rebuilds everything from scratch.
"""
import json
import time
from datetime import datetime, timezone

from ledger import config
from ledger.chunk import chunk_document, context_header
from ledger.parse import html_to_text


def make_qdrant_client():
    """Connect to Qdrant: a local folder by default, or a server if QDRANT_URL is set."""
    from qdrant_client import QdrantClient
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    # Local mode: Qdrant stores everything in a folder. Simple, no server,
    # BUT only one program can have the folder open at a time.
    return QdrantClient(path=str(config.QDRANT_PATH))


def parse_and_chunk_all() -> list[dict]:
    """Parse every downloaded filing and return all chunks as plain dicts."""
    config.TEXT_DIR.mkdir(parents=True, exist_ok=True)
    meta_files = sorted(config.RAW_DIR.glob("*.json"))
    if not meta_files:
        raise SystemExit("No filings in data/raw/. Run `uv run python -m ledger.edgar` first.")

    all_chunks: list[dict] = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        html_path = meta_path.with_suffix(".html")
        started = time.perf_counter()

        text = html_to_text(html_path.read_text(encoding="utf-8", errors="replace"))
        # Save the clean text. Citations point at character positions in THIS file.
        (config.TEXT_DIR / f"{meta_path.stem}.txt").write_text(text, encoding="utf-8")

        chunks = chunk_document(text, meta)
        all_chunks.extend(c.to_dict() for c in chunks)

        tables = text.count("[TABLE]")
        sections = len({c.section for c in chunks})
        print(f"  {meta_path.stem:<18} {len(text):>9,} chars  {len(chunks):>4} chunks  "
              f"{tables:>4} tables  {sections:>2} sections  ({time.perf_counter() - started:.1f}s)")
    return all_chunks


def build() -> None:
    # These imports are slow (they load PyTorch), so they happen inside the
    # function rather than at the top of the file.
    from qdrant_client import models
    from sentence_transformers import SentenceTransformer

    print("1/4 Parsing and chunking filings...")
    chunks = parse_and_chunk_all()

    print(f"\n2/4 Writing {len(chunks):,} chunks to {config.CHUNKS_PATH.name}...")
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")   # one JSON object per line

    print(f"\n3/4 Embedding with {config.EMBED_MODEL} (first run downloads the model)...")
    model = SentenceTransformer(config.EMBED_MODEL)
    # What we embed = context header + chunk text. normalize_embeddings=True
    # scales every vector to length 1, so cosine similarity works correctly.
    texts = [context_header(c) + "\n" + c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True)
    dimension = vectors.shape[1]      # 384 for bge-small

    print(f"\n4/4 Storing vectors in Qdrant collection '{config.QDRANT_COLLECTION}'...")
    client = make_qdrant_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)     # rebuild from scratch
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    # A "point" = one vector + an id + a "payload" (extra fields we can filter on).
    batch_size = 256
    for i in range(0, len(chunks), batch_size):
        points = [
            models.PointStruct(
                id=i + j,                              # Qdrant ids must be ints or UUIDs
                vector=vectors[i + j].tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "ticker": chunk["ticker"],
                    "fiscal_year": chunk["fiscal_year"],
                    "section": chunk["section"],
                },
            )
            for j, chunk in enumerate(chunks[i:i + batch_size])
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    client.close()

    # Record HOW this index was built. If you later change the embedding
    # model and forget to rebuild, this file tells you why search broke.
    info = {"embed_model": config.EMBED_MODEL, "dimension": int(dimension),
            "chunks": len(chunks), "built_at": datetime.now(timezone.utc).isoformat()}
    (config.DATA_DIR / "index_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDone. {len(chunks):,} chunks indexed.")


if __name__ == "__main__":
    build()
