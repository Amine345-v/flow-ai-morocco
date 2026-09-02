import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger


class HermesMemoryStore:
    """
    SQLite-backed FTS5 persistent memory store for FlowLang / JOL Studio.
    Provides session tracking, procedural memory indexing, cross-flow recall,
    and full-text search capability inspired by Nous Research's Hermes Agent.
    """

    def __init__(self, db_path: str = "./.flowlang_state/hermes_memory.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize relational and FTS5 search tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flow_sessions (
                    session_id TEXT PRIMARY KEY,
                    flow_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)

            # Structured memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flow_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    flow_name TEXT NOT NULL,
                    checkpoint_name TEXT,
                    team_name TEXT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES flow_sessions(session_id)
                )
            """)

            # Try initializing FTS5 search table
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS flow_memories_fts USING fts5(
                        title,
                        content,
                        tags,
                        category,
                        content='flow_memories',
                        content_rowid='id'
                    )
                """)
                # Triggers to keep FTS table synchronized with flow_memories
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS flow_memories_ai AFTER INSERT ON flow_memories BEGIN
                        INSERT INTO flow_memories_fts(rowid, title, content, tags, category)
                        VALUES (new.id, new.title, new.content, new.tags, new.category);
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS flow_memories_ad AFTER DELETE ON flow_memories BEGIN
                        INSERT INTO flow_memories_fts(flow_memories_fts, rowid, title, content, tags, category)
                        VALUES('delete', old.id, old.title, old.content, old.tags, old.category);
                    END;
                """)
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS flow_memories_au AFTER UPDATE ON flow_memories BEGIN
                        INSERT INTO flow_memories_fts(flow_memories_fts, rowid, title, content, tags, category)
                        VALUES('delete', old.id, old.title, old.content, old.tags, old.category);
                        INSERT INTO flow_memories_fts(rowid, title, content, tags, category)
                        VALUES (new.id, new.title, new.content, new.tags, new.category);
                    END;
                """)
                self._has_fts = True
            except sqlite3.OperationalError as e:
                logger.warning(f"SQLite FTS5 extension not available or failed: {e}. Falling back to LIKE queries.")
                self._has_fts = False

            conn.commit()

    def start_session(self, session_id: str, flow_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Record the start of a flow session."""
        timestamp = datetime.now().isoformat()
        metadata_str = json.dumps(metadata or {})
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO flow_sessions (session_id, flow_name, timestamp, status, metadata_json)
                VALUES (?, ?, ?, 'RUNNING', ?)
            """, (session_id, flow_name, timestamp, metadata_str))
            conn.commit()

    def end_session(self, session_id: str, status: str = "COMPLETED"):
        """Record session completion."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE flow_sessions SET status = ? WHERE session_id = ?
            """, (status, session_id))
            conn.commit()

    def add_memory(
        self,
        category: str,
        title: str,
        content: str,
        flow_name: str,
        checkpoint_name: Optional[str] = None,
        team_name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Add a procedural memory, checkpoint summary, error resolution, or user preference.
        Categories: 'procedural', 'report', 'error_resolution', 'user_preference'
        """
        timestamp = datetime.now().isoformat()
        tags_str = ",".join(tags) if tags else ""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO flow_memories (session_id, flow_name, checkpoint_name, team_name, category, title, content, tags, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, flow_name, checkpoint_name, team_name, category, title, content, tags_str, timestamp))
            conn.commit()
            memory_id = cursor.lastrowid
            logger.info(f"💾 [HermesMemory] Added memory #{memory_id} [{category}] '{title}' for team '{team_name}'")
            return memory_id

    def search_memories(
        self,
        query: str,
        category: Optional[str] = None,
        flow_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search memories using FTS5 syntax or keyword matching."""
        if not query or not query.strip():
            return self.get_latest_memories(category=category, flow_name=flow_name, limit=limit)

        clean_query = query.replace("'", "''").strip()
        results = []

        with self._get_connection() as conn:
            if self._has_fts:
                try:
                    sql = """
                        SELECT m.* FROM flow_memories m
                        JOIN flow_memories_fts fts ON m.id = fts.rowid
                        WHERE flow_memories_fts MATCH ?
                    """
                    params = [clean_query]

                    if category:
                        sql += " AND m.category = ?"
                        params.append(category)
                    if flow_name:
                        sql += " AND m.flow_name = ?"
                        params.append(flow_name)

                    sql += " ORDER BY m.id DESC LIMIT ?"
                    params.append(limit)

                    cursor = conn.execute(sql, params)
                    results = [dict(row) for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    results = []

            if not results:
                # Fallback to standard SQL LIKE search
                sql = "SELECT * FROM flow_memories WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
                like_term = f"%{clean_query}%"
                params = [like_term, like_term, like_term]

                if category:
                    sql += " AND category = ?"
                    params.append(category)
                if flow_name:
                    sql += " AND flow_name = ?"
                    params.append(flow_name)

                sql += " ORDER BY id DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_latest_memories(
        self,
        category: Optional[str] = None,
        flow_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve recent memories sorted by newest first."""
        with self._get_connection() as conn:
            sql = "SELECT * FROM flow_memories WHERE 1=1"
            params = []
            if category:
                sql += " AND category = ?"
                params.append(category)
            if flow_name:
                sql += " AND flow_name = ?"
                params.append(flow_name)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def format_memory_prompt_context(self, query: str, flow_name: str, limit: int = 3) -> str:
        """Formats relevant stored memories into markdown for LLM prompt augmentation."""
        memories = self.search_memories(query=query, flow_name=flow_name, limit=limit)
        if not memories:
            memories = self.get_latest_memories(category="procedural", limit=limit)

        if not memories:
            return ""

        lines = ["\n🧠 [HERMES PERSISTENT MEMORY & PROCEDURAL KNOWLEDGE]"]
        for mem in memories:
            lines.append(f"  • Category: {mem['category'].upper()} | Team: {mem.get('team_name', 'N/A')}")
            lines.append(f"    Title: {mem['title']}")
            lines.append(f"    Snippet: {mem['content'][:300]}...")
        lines.append("")
        return "\n".join(lines)
