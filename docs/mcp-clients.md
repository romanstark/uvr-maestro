# MCP client examples

The repository includes these checkout examples:

| Client or example | File |
| --- | --- |
| Generic agent example | `.agents/mcp_config.json` |
| Claude example | `.claude/mcp.json` |
| Codex project configuration | `.codex/config.toml` |
| Cursor project configuration | `.cursor/mcp.json` |
| VS Code project configuration | `.vscode/mcp.json` |

Each example starts `uv run --managed-python --python 3.12 uvr-maestro` with the checkout root as its working
directory. If the client starts elsewhere, use
`uv run --managed-python --python 3.12 --directory <absolute-checkout-path> uvr-maestro`, or replace the command
with the absolute installed `uvr-maestro` executable. Set runtime/model path
overrides as described in the README when using a custom local installation.
Provision the engine separately with `uvr-maestro-setup --confirm`.
The Git-based root `.mcp.json` uses `uvx` with the same interpreter options.
Managed Python avoids Microsoft Store Python's private AppData view on Windows.

Generic and Claude example files are templates to copy into the client's actual
configuration. For Claude Code's project scope, use the root `.mcp.json` when
choosing checkout execution; its Git-based example requires repository access.
Use an untracked `mcp.local.json` for machine-specific runtime paths. Portable
examples contain no private paths or credentials.

Codex uses `[mcp_servers.<name>]` in its project `.codex/config.toml` for a trusted
project. See the [official MCP configuration documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
VS Code uses `servers` with a `stdio` type; the other JSON examples use
`mcpServers`.

The five skills live once in `src/uvr_maestro/skills/`. They are included in the
wheel, available through `uvr_skills` and `uvr://skill/{name}`, and exportable:

```text
uvr-maestro-skills export --output <your-client-skill-directory>
```

Export refuses to overwrite existing skills. A client-native skill installation
is separate from configuring the MCP connection.
