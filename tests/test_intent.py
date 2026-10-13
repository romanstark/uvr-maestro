"""Intent ranker: producer keywords map deterministically to catalog rows."""

from __future__ import annotations

from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.engine.intent import rank_intent
from uvr_maestro.registry import Registry


def test_intent_vocals_returns_vocal_row() -> None:
    registry = Registry.load_bundled(CATALOG_DIR)
    rows = rank_intent("I need the vocals", registry=registry)
    assert rows
    top = rows[0]
    assert top[0].primary_stem.value == "vocals"


def test_intent_de_echo() -> None:
    registry = Registry.load_bundled(CATALOG_DIR)
    rows = rank_intent("de-echo the recording", registry=registry)
    assert rows
    assert any("De-Echo" in c[0].display_name for c in rows)


def test_intent_karaoke() -> None:
    registry = Registry.load_bundled(CATALOG_DIR)
    rows = rank_intent("make a karaoke stem", registry=registry)
    assert rows
    assert rows[0][0].primary_stem.value == "karaoke"


def test_intent_empty_returns_none() -> None:
    registry = Registry.load_bundled(CATALOG_DIR)
    rows = rank_intent("hello world", registry=registry)
    assert rows == []
