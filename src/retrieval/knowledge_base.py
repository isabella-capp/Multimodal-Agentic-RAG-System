import sqlite3
import threading


class KnowledgeBase:
    """Read-only encyclopedic KB backed by a SQLite file.

    Build the file once with ``src/retrieval/build_kb_sqlite.py``. Lookups hit
    the disk on demand, so the 15 GB KB is never loaded into memory.

    Thread-safe: each thread gets its own connection (SQLite connections cannot
    be shared across threads for concurrent queries). The DB is opened read-only
    and immutable, so concurrent readers are fine.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._uri = f"file:{db_path}?mode=ro&immutable=1"
        self._local = threading.local()
        self._conn()  # fail fast if the file is missing
        print(f"Knowledge Base ready (SQLite): {db_path}")

    def get_paragraphs_by_url(self, wiki_url: str) -> list[str]:
        """Return the non-empty section texts for a Wikipedia URL, in order."""
        rows = self._conn().execute(
            "SELECT text FROM paragraphs WHERE url = ? ORDER BY section_idx",
            (wiki_url,),
        ).fetchall()
        return [r[0] for r in rows]

    def __contains__(self, wiki_url: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM articles WHERE url = ? LIMIT 1", (wiki_url,)
        ).fetchone()
        return row is not None

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn
