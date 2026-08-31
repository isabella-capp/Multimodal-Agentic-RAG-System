"""Build the SQLite KB from the 15 GB encyclopedic_kb_wiki.json.

The source is a dict ``{url: {section_texts, section_titles, title,
image_urls, ...}}`` with ~2M articles, streamed with ijson (constant memory)
into two tables: ``articles`` (metadata) and ``paragraphs`` (one row per
non-empty section text, linked by url). The text lives only in ``paragraphs``.

A final pass adds the name-lookup tables (``aliases`` + ``titles_fts``) that
``KnowledgeBase.lookup_articles`` queries, so one run produces a KB the agent
can use as-is. Use ``--index-only`` to rebuild just those on an existing KB.

``--paragraphs-fts`` adds ``paragraphs_fts``, a full-text index over paragraph
*text*, so a question can find an article without anyone naming it. It writes
into the KB itself: one file stays the single source of truth for retrieval.

Run it when nothing else is reading the KB. It takes a write lock on an 18.6 GB
file the whole group reads, and readers must not have it open with
``immutable=1`` — that flag promises SQLite the bytes never change, which stops
being true here. ``KnowledgeBase`` therefore opens read-only *without* it.
"""

import argparse
import json
import os
import sqlite3
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

import ijson
from tqdm import tqdm

from retrieval.knowledge_base import alias_forms, title_tokens

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
KB_JSON_PATH = f"{BASE_FOLDER}/encyclopedic_kb_wiki.json"
KB_DB_PATH = f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"

TOTAL_ARTICLES = 2_004_561  # progress-bar hint only; an estimate is fine
BATCH_SIZE = 5000


def create_schema(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS articles;
        DROP TABLE IF EXISTS paragraphs;
        CREATE TABLE articles (
            url            TEXT PRIMARY KEY,
            title          TEXT,
            section_titles TEXT,
            image_urls     TEXT
        );
        CREATE TABLE paragraphs (
            id            INTEGER PRIMARY KEY,
            url           TEXT NOT NULL,
            section_idx   INTEGER,
            section_title TEXT,
            text          TEXT
        );
        """
    )


NAME_INDEX_SCHEMA = """
DROP TABLE IF EXISTS aliases;
DROP TABLE IF EXISTS titles_fts;
CREATE TABLE aliases (
    alias TEXT NOT NULL,
    url   TEXT NOT NULL
);
CREATE VIRTUAL TABLE titles_fts USING fts5(tokens, url UNINDEXED, tokenize='unicode61');
"""


def build_name_index(conn):
    """Add the name → article lookup tables over the existing ``articles``.

    Normalisation must match ``KnowledgeBase`` exactly, hence the shared helpers:
    a mismatch between build time and query time fails silently.
    """
    print("Reading articles for the name index …")
    articles = conn.execute("SELECT url, title FROM articles").fetchall()
    conn.executescript(NAME_INDEX_SCHEMA)

    alias_rows, fts_rows, n_aliases = [], [], 0

    def flush():
        conn.executemany("INSERT INTO aliases VALUES (?,?)", alias_rows)
        conn.executemany("INSERT INTO titles_fts(tokens, url) VALUES (?,?)", fts_rows)
        alias_rows.clear()
        fts_rows.clear()

    for url, title in tqdm(articles, desc="Name index"):
        for alias in alias_forms(title):
            alias_rows.append((alias, url))
            n_aliases += 1
        fts_rows.append((" ".join(sorted(title_tokens(title))), url))
        if len(fts_rows) >= BATCH_SIZE * 10:
            flush()
    flush()

    print("Creating index on aliases(alias) …")
    conn.execute("CREATE INDEX idx_aliases_alias ON aliases(alias)")
    conn.commit()
    print(f"Name index: {len(articles)} articles, {n_aliases} aliases")


def build_paragraph_fts(db_path, overwrite):
    """Index the text of every paragraph, in the KB, next to the other indexes.

    The two channels we had both run through the model: the image index needs
    the entity to have a photograph (40.6% recall@20) and the name lookup needs
    the model to name it, which Qwen3-VL-8B manages 11.6% of the time. This one
    does not — the question itself is the query.

    ``content=''`` keeps the table contentless, so the text is not stored a
    second time and only the inverted index is added. Rows are keyed by
    ``paragraphs.id``, so a hit joins straight back to its article.

    Pragmas are the safe ones on purpose: a crash during this must leave the
    18.6 GB KB exactly as it was, which is worth more than a faster build.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")

    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='paragraphs_fts'").fetchone()
    if exists and not overwrite:
        raise SystemExit("paragraphs_fts already exists (use --overwrite to rebuild)")
    if exists:
        conn.execute("DROP TABLE paragraphs_fts")
        conn.commit()

    total = conn.execute("SELECT max(id) FROM paragraphs").fetchone()[0] or 0
    # porter stemming: questions say "sequenced", articles say "sequencing"
    conn.execute("CREATE VIRTUAL TABLE paragraphs_fts USING fts5"
                 "(text, content='', tokenize='porter unicode61')")

    batch, n = [], 0
    read = conn.execute("SELECT id, text FROM paragraphs")
    write = conn.cursor()
    for pid, text in tqdm(read, total=total, desc="Paragraphs"):
        if not text:
            continue
        batch.append((pid, text))
        n += 1
        if len(batch) >= BATCH_SIZE * 10:
            write.executemany("INSERT INTO paragraphs_fts(rowid, text) VALUES (?,?)", batch)
            batch.clear()
    if batch:
        write.executemany("INSERT INTO paragraphs_fts(rowid, text) VALUES (?,?)", batch)
    conn.commit()

    print("Optimising the FTS index …")
    conn.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES ('optimize')")
    conn.commit()
    conn.close()
    print(f"Done: {n} paragraphs indexed into {db_path} "
          f"({os.path.getsize(db_path) / 2**30:.1f} GB total)")


def iter_rows(json_path):
    """Yield ``(article_row, paragraph_rows)`` for each article, streaming."""
    with open(json_path, "rb") as f:
        for url, art in ijson.kvitems(f, ""):
            section_titles = art.get("section_titles") or []
            section_texts = art.get("section_texts") or []
            image_urls = art.get("image_urls") or []

            article_row = (
                url,
                art.get("title", ""),
                json.dumps(section_titles, ensure_ascii=False),
                json.dumps(image_urls, ensure_ascii=False),
            )

            para_rows = []
            for idx, text in enumerate(section_texts):
                if not text or not text.strip():
                    continue
                title = section_titles[idx] if idx < len(section_titles) else None
                para_rows.append((url, idx, title, text))

            yield article_row, para_rows


def build(json_path, db_path, overwrite):
    if os.path.exists(db_path):
        if not overwrite:
            raise SystemExit(f"{db_path} already exists (use --overwrite to rebuild)")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    # Fast, unsafe pragmas: acceptable for a one-shot build of static data.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")  # ~256 MB page cache
    create_schema(conn)

    art_batch, para_batch = [], []
    n_articles = n_paragraphs = 0

    def flush():
        conn.executemany("INSERT OR IGNORE INTO articles VALUES (?,?,?,?)", art_batch)
        conn.executemany(
            "INSERT INTO paragraphs (url, section_idx, section_title, text) "
            "VALUES (?,?,?,?)",
            para_batch,
        )
        art_batch.clear()
        para_batch.clear()

    for article_row, para_rows in tqdm(
        iter_rows(json_path), total=TOTAL_ARTICLES, desc="Articles"
    ):
        art_batch.append(article_row)
        para_batch.extend(para_rows)
        n_articles += 1
        n_paragraphs += len(para_rows)
        if len(art_batch) >= BATCH_SIZE:
            flush()

    if art_batch:
        flush()
    conn.commit()

    print("Creating index on paragraphs(url) …")
    conn.execute("CREATE INDEX idx_paragraphs_url ON paragraphs(url)")
    conn.commit()

    build_name_index(conn)

    print("Optimising (ANALYZE) …")
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    print(f"Done: {n_articles} articles, {n_paragraphs} paragraphs -> {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the SQLite KB from encyclopedic_kb_wiki.json"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the .db already exists.",
    )
    parser.add_argument(
        "--paragraphs-fts",
        action="store_true",
        help="Add the full-text index over paragraph text to the KB.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild only the name-lookup tables on an existing KB.",
    )
    args = parser.parse_args()

    if args.paragraphs_fts:
        build_paragraph_fts(KB_DB_PATH, args.overwrite)
        return

    if args.index_only:
        # Safe pragmas: a crash must leave the existing KB intact.
        conn = sqlite3.connect(KB_DB_PATH)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        build_name_index(conn)
        conn.execute("ANALYZE")
        conn.commit()
        conn.close()
        return

    build(KB_JSON_PATH, KB_DB_PATH, args.overwrite)


if __name__ == "__main__":
    main()
