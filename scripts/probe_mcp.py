"""Probe the installed package through real MCP stdio, optionally running audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl


def payload(result: Any) -> dict[str, Any]:
    """Read structured MCP data with a text fallback."""
    if result.structuredContent:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def probe(args: argparse.Namespace) -> None:
    """Initialize, discover and call the installed server outside its checkout."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["UVR_MAESTRO_DATA_DIR"] = str(args.data.resolve())
    if args.runtime:
        environment["UVR_MAESTRO_RUNTIME_DIR"] = str(args.runtime.resolve())
        environment.pop("UVR_MAESTRO_ENGINE_PYTHON", None)
    command = StdioServerParameters(
        command=str(args.python),
        args=["-m", "uvr_maestro.server"],
        env=environment,
        cwd=str(args.data.parent.resolve()),
    )
    async with (
        stdio_client(command) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        initialization = await session.initialize()
        tools = await session.list_tools()
        assert len(tools.tools) == 34
        status = payload(await session.call_tool("uvr_status"))
        assert status["success"]
        listed = payload(await session.call_tool("list_models", {"arch": "mdx"}))
        assert listed["success"] and listed["rows"]
        resource = await session.read_resource(AnyUrl("uvr://catalog"))
        assert resource.contents
        prompts = await session.list_prompts()
        assert len(prompts.prompts) == 2
        skills = payload(await session.call_tool("uvr_skills"))
        assert len(skills["skills"]) == 5
        await session.read_resource(AnyUrl("uvr://skill/uvr-separation"))
        refusal = payload(
            await session.call_tool(
                "set_settings",
                {
                    "values": {"device": "cpu"},
                    "confirm": False,
                },
            )
        )
        assert refusal["blocked"] and not refusal["success"]
        print(
            json.dumps(
                {
                    "initialized": initialization.serverInfo.name,
                    "tools": len(tools.tools),
                    "prompts": len(prompts.prompts),
                    "skills": len(skills["skills"]),
                    "engine_ready": status["engine"]["ready"],
                }
            ),
            flush=True,
        )
        if args.audio:
            started = payload(
                await session.call_tool(
                    "start_separation",
                    {
                        "audio_path": str(args.audio.resolve()),
                        "model": "Kim Vocal 2",
                        "output_dir": str((args.data / "output").resolve()),
                        "options": {"device": "cuda"},
                    },
                )
            )
            assert started["success"], started
            while True:
                job = payload(await session.call_tool("get_job", {"job_id": started["id"]}))
                if job["state"] in ("succeeded", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
            assert job["state"] == "succeeded", job
            assert Path(job["manifest"]).is_file()
            assert job["result"]["device"] == "cpu"
            assert job["result"]["fallback_reason"]
            assert len(job["result"]["outputs"]) == 2
            (args.data / "mcp-result.json").write_text(json.dumps(job, indent=2))
            print(
                json.dumps(
                    {
                        "separation": job["state"],
                        "device": job["result"]["device"],
                        "fallback_reason": job["result"]["fallback_reason"],
                        "outputs": len(job["result"]["outputs"]),
                    }
                ),
                flush=True,
            )


def main() -> None:
    """Choose an installed interpreter and an isolated data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    # Resolving a venv's launcher symlink discards that environment's packages.
    args.python = args.python.expanduser().absolute()
    args.data = args.data.resolve()
    if args.runtime:
        args.runtime = args.runtime.resolve()
    if args.audio:
        args.audio = args.audio.resolve()
    os.chdir(args.data.parent)
    asyncio.run(probe(args))


if __name__ == "__main__":
    main()
