import re
import sqlite3
import threading
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9 ]")
_SPACE = re.compile(r"\s+")
_PAREN = re.compile(r"\s*\([^)]*\)")
_STOP = {"the", "a", "an", "of", "in", "and", "on", "at"}

_MIN_JACCARD = 0.5
_FUZZY_CANDIDATES = 200  # BM25 shortlist, re-scored exactly below


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return _SPACE.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def alias_forms(title: str) -> list[str]:
    """Surface forms for a title, most specific first.

    Ordered, not a set: lookup tries them in turn, and set iteration order over
    strings varies between processes, which would make matches non-reproducible.
    """
    base = normalize(title)
    forms = [base, normalize(_PAREN.sub("", title))]
    forms += [f[4:] for f in forms if f.startswith("the ")]
    return list(dict.fromkeys(f for f in forms if f))


def title_tokens(title: str) -> set[str]:
    tokens: set[str] = set()
    for alias in alias_forms(title):
        tokens |= set(alias.split())
    return tokens - _STOP


class KnowledgeBase:
    """Read-only encyclopedic KB backed by a SQLite file.

    Build the file once with ``src/retrieval/build_kb_sqlite.py``, then add the
    name-lookup tables with ``src/retrieval/build_title_index.py``. Lookups hit
    the disk on demand, so the 20 GB KB is never loaded into memory.

    ``lookup_articles`` resolves an entity NAME to its article by string matching
    — deliberately not by embedding: the EVA-CLIP text tower is misaligned with
    the image index (0% recall@50 even given the ground-truth title), and exact
    matching is anyway more precise than similarity for near-identical names.

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

    def lookup_articles(self, name: str, limit: int = 5) -> list[dict]:
        """Articles whose title matches ``name``; an exact match returns one."""
        query = normalize(name)
        if not query:
            return []
        conn = self._conn()
        for form in alias_forms(name):
            row = conn.execute(
                "SELECT a.title, a.url FROM aliases al "
                "JOIN articles a ON a.url = al.url "
                "WHERE al.alias = ? ORDER BY al.rowid LIMIT 1",
                (form,),
            ).fetchone()
            if row:
                return [{"title": row[0], "wiki_url": row[1], "match": "exact"}]
        return self._fuzzy(query, limit)

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

    def _fuzzy(self, query: str, limit: int) -> list[dict]:
        qtokens = set(query.split()) - _STOP
        if not qtokens:
            return []
        rows = self._conn().execute(
            "SELECT a.title, f.url, f.tokens FROM titles_fts f "
            "JOIN articles a ON a.url = f.url "
            "WHERE titles_fts MATCH ? ORDER BY bm25(titles_fts) LIMIT ?",
            (" OR ".join(sorted(qtokens)), _FUZZY_CANDIDATES),
        ).fetchall()

        ranked = []
        for title, url, tokens in rows:
            ttokens = set(tokens.split())
            shared = len(qtokens & ttokens)
            if shared < 2:
                continue
            union = len(qtokens) + len(ttokens) - shared
            score = shared / union if union else 0.0
            if score >= _MIN_JACCARD or (shared == len(ttokens) and len(ttokens) >= 2):
                ranked.append((score, len(ttokens), title, url))

        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [{"title": t, "wiki_url": u, "match": "fuzzy"} for _, _, t, u in ranked[:limit]]
