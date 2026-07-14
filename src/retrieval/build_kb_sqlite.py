"""Build the SQLite KB from the 15 GB encyclopedic_kb_wiki.json.

The source is a dict ``{url: {section_texts, section_titles, title,
image_urls, ...}}`` with ~2M articles, streamed with ijson (constant memory)
into two tables: ``articles`` (metadata) and ``paragraphs`` (one row per
non-empty section text, linked by url). The text lives only in ``paragraphs``.
"""

import argparse
import json
import os
import sqlite3

import ijson
from tqdm import tqdm

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
    args = parser.parse_args()
    build(KB_JSON_PATH, KB_DB_PATH, args.overwrite)


if __name__ == "__main__":
    main()
