"""
src/evidence_retrieval/postgres_adapter.py
------------------------------------------
PostgreSQL and Local Relational Storage Adapter for Historical Inscriptions.

As specified in EpiRecover PPT Slide 13:
  Software requirements: "PostgreSQL (historical corpus)"
  and Slide 10:
  "BharatSHRI (ASI) 13,986 metadata records, 9,967 transcripts, 10,116 digitised estampages"
  "SIDDHAM EpiDoc (TEI XML) texts + metadata"

Architecture:
  - Manages structured schemas for epigraphic records:
      * inscriptions (id, language, script, transcription, historical_period, region, monument, citation, is_synthetic)
      * review_decisions (id, inscription_id, accepted_candidate, reviewer, timestamp, notes)
  - Supports live PostgreSQL databases via psycopg2 or SQLAlchemy.
  - Automatically provides SQLite / in-memory fallback for local development and offline testing,
    ensuring zero friction when a live PostgreSQL daemon is not running.

EpiRecover, 2026.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS inscriptions (
    id VARCHAR(64) PRIMARY KEY,
    language VARCHAR(64) NOT NULL,
    script VARCHAR(64) NOT NULL,
    transcription TEXT NOT NULL,
    historical_period VARCHAR(128),
    region VARCHAR(128),
    monument VARCHAR(128),
    dynasty VARCHAR(128),
    source_citation TEXT,
    is_synthetic BOOLEAN DEFAULT FALSE,
    provenance_notes TEXT
);

CREATE TABLE IF NOT EXISTS review_decisions (
    decision_id VARCHAR(64) PRIMARY KEY,
    inscription_id VARCHAR(64) NOT NULL,
    candidate_id VARCHAR(64),
    candidate_text TEXT NOT NULL,
    status VARCHAR(32) NOT NULL, -- 'ACCEPTED', 'EDITED', 'REJECTED'
    reviewer_notes TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class InscriptionRecord:
    id: str
    language: str
    script: str
    transcription: str
    historical_period: Optional[str] = None
    region: Optional[str] = None
    monument: Optional[str] = None
    dynasty: Optional[str] = None
    source_citation: Optional[str] = None
    is_synthetic: bool = False
    provenance_notes: Optional[str] = None


class PostgresCorpusAdapter:
    """
    Manages connection and queries to the historical corpus database.
    Seamlessly switches between PostgreSQL and SQLite local fallback.
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        sqlite_fallback_path: str = "data/epirecover_corpus.db",
    ) -> None:
        self.connection_url = connection_url or os.getenv("DATABASE_URL")
        self.sqlite_path = sqlite_fallback_path
        self.is_postgres = False
        self._conn = None

        if self.connection_url and "postgresql" in self.connection_url:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self.connection_url)
                self.is_postgres = True
                log.info("Connected to live PostgreSQL corpus database.")
            except Exception as e:
                log.warning("PostgreSQL connection failed (%s); using local SQLite database at %s.", e, self.sqlite_path)
                self._init_sqlite()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self.is_postgres = False
        with self._conn:
            self._conn.executescript(CREATE_TABLES_SQL)
        log.info("PostgresCorpusAdapter initialized with SQLite local storage at %s", self.sqlite_path)

    def insert_record(self, record: InscriptionRecord) -> None:
        """Insert or replace an inscription record."""
        sql = """
        INSERT OR REPLACE INTO inscriptions (
            id, language, script, transcription, historical_period,
            region, monument, dynasty, source_citation, is_synthetic, provenance_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """ if not self.is_postgres else """
        INSERT INTO inscriptions (
            id, language, script, transcription, historical_period,
            region, monument, dynasty, source_citation, is_synthetic, provenance_notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            transcription = EXCLUDED.transcription,
            source_citation = EXCLUDED.source_citation;
        """
        params = (
            record.id, record.language, record.script, record.transcription,
            record.historical_period, record.region, record.monument,
            record.dynasty, record.source_citation, record.is_synthetic,
            record.provenance_notes,
        )
        cur = self._conn.cursor()
        cur.execute(sql, params)
        self._conn.commit()

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Fetch record by ID."""
        sql = "SELECT * FROM inscriptions WHERE id = ?" if not self.is_postgres else "SELECT * FROM inscriptions WHERE id = %s"
        cur = self._conn.cursor()
        cur.execute(sql, (record_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def search_records(
        self,
        query: str,
        script: Optional[str] = None,
        language: str = "Tamil",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search records matching criteria."""
        conditions = ["language = ?"] if not self.is_postgres else ["language = %s"]
        params = [language]

        if script:
            conditions.append("script = ?" if not self.is_postgres else "script = %s")
            params.append(script)

        sql = f"SELECT * FROM inscriptions WHERE {' AND '.join(conditions)} LIMIT {limit}"
        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def record_count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM inscriptions")
        res = cur.fetchone()
        return res[0] if res else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
