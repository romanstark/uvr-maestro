"""Read or export the client-neutral skills included in every wheel."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    """Export skills without overwriting existing folders implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).parent / "skills"
    for path in sorted(source.iterdir()):
        if (args.output / path.name).exists():
            raise FileExistsError(f"skill already exists: {args.output / path.name}")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        shutil.copytree(path, args.output / path.name)
    print(f"Exported skills to {args.output.resolve()}")


if __name__ == "__main__":
    main()
