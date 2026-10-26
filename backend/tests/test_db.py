"""Catalog plumbing: the additive migrations and the per-shoot export memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from culler.db import Store


def test_old_catalog_gains_the_export_columns(tmp_path: Path):
    """A catalog written by an older build must keep working, not crash."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE libraries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_path TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            settings_json TEXT
        );
        INSERT INTO libraries (root_path, name, created_at)
        VALUES ('/photos/shoot', 'shoot', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    store = Store(path)
    assert store.get_library(1)["export_mode"] is None
    store.update_library_export(1, "copy", "/photos/delivery")

    library = store.get_library(1)
    assert library["export_mode"] == "copy"
    assert library["export_dest"] == "/photos/delivery"


def test_export_memory_is_per_library(tmp_path: Path):
    store = Store(tmp_path / "catalog.db")
    first = store.create_or_get_library(str(tmp_path / "a"))["id"]
    second = store.create_or_get_library(str(tmp_path / "b"))["id"]

    store.update_library_export(first, "copy", "  /photos/delivery  ")
    assert store.get_library(first)["export_dest"] == "/photos/delivery"
    assert store.get_library(second)["export_mode"] is None

    # A sidecar export has no destination: the mode moves on, the folder stays.
    store.update_library_export(first, "sidecar_all", "")
    assert store.get_library(first)["export_mode"] == "sidecar_all"
    assert store.get_library(first)["export_dest"] == "/photos/delivery"
