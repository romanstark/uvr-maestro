"""Release claims checked against the actual public package surface."""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from pathlib import Path

import yaml

from uvr_maestro import __version__, server
from uvr_maestro.catalog import CATALOG_DIR
from uvr_maestro.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def test_version_consistency() -> None:
    """Keep package, metadata and optional client plugin on one release version."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))
    assert project["project"]["version"] == __version__
    plugin = ROOT / ".claude-plugin/plugin.json"
    if plugin.exists():
        assert json.loads(plugin.read_text())["version"] == __version__


def test_readme_counts_and_names_match_public_surface() -> None:
    """Reject stale tool counts, examples and invented names in the README."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    tools = asyncio.run(server.mcp.list_tools())
    prompts = asyncio.run(server.mcp.list_prompts())
    assert f"**{len(tools)} tools**" in readme
    assert f"**{len(Registry.load_bundled().all())} models**" in readme
    names = {tool.name for tool in tools} | {prompt.name for prompt in prompts}
    for name in names:
        assert f"`{name}`" in readme, name
    assert len(asyncio.run(server.mcp.list_resources())) == 3
    assert len(asyncio.run(server.mcp.list_resource_templates())) == 1


def test_shipped_assets_and_notices_exist() -> None:
    """Runtime setup and skills remain available from a wheel."""
    package = Path(server.__file__).parent
    assert (package / "engine/worker.py").is_file()
    assert (package / "engine/requirements.txt").is_file()
    assert (package / "engine/diffq_bitpack.py").is_file()
    assert (CATALOG_DIR / "vr_metadata.json").is_file()
    assert (CATALOG_DIR / "mdx_metadata.json").is_file()
    assert len((ROOT / "LICENSE").read_text().splitlines()) > 600


def test_skills_frontmatter_and_real_tool_references() -> None:
    """Check portable skill frontmatter and references against actual registration."""
    folder = Path(server.__file__).parent / "skills"
    tools = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    paths = sorted(folder.glob("*/SKILL.md"))
    assert len(paths) == 5
    for path in paths:
        text = path.read_text()
        match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
        assert match is not None
        data = yaml.safe_load(match.group(1))
        assert data["name"] == path.parent.name
        assert 40 < len(data["description"]) < 220
        assert "uvr-maestro-setup" not in tools
        referenced = set(re.findall(r"`([a-z][a-z_]+)`", text))
        assert referenced <= tools, referenced - tools


def test_relative_readme_links_exist() -> None:
    """A release must not link to local-only plans or missing artifacts."""
    text = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if not target.startswith(("https://", "#")):
            assert (ROOT / target).is_file(), target
