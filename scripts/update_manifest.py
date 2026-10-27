#!/usr/bin/env python3
"""Write the release's `update.json` — the manifest the in-app updater reads.

Published as a release asset, it is what the app fetches from the stable
``.../releases/latest/download/update.json`` URL when the user asks it to check
for updates. It carries the version, the ``.dmg`` URL and that file's SHA-256.

The hash is never computed here from scratch: it is read out of the release's
own ``SHA256SUMS``, so the manifest and the checksum file cannot disagree about
what was published. The payload is then run through the app's own validator
(``culler/update.py``) before it is written, so a release cannot publish a
manifest the app would refuse to read.

    python3 scripts/update_manifest.py \\
        --version 0.9.0 --tag v0.9.0 --repo Salcedoandrew/cullr \\
        --sums dist/SHA256SUMS --dmg dist/Cullr-0.9.0-arm64.dmg \\
        --out dist/update.json

Standard library only: it runs before/independently of the project venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The app's own manifest reader/validator, so the published shape is provably
# one the updater accepts. culler/update.py imports nothing outside the standard
# library, which is what makes this usable from a plain python3.
sys.path.insert(0, str(ROOT / "backend"))

from culler.update import UpdateError, is_newer, validate_manifest  # noqa: E402

DIGEST_RE = re.compile(r"^(?P<sha>[0-9a-fA-F]{64})\s+\*?(?P<name>.+?)\s*$")


def read_sums(path: Path) -> dict[str, str]:
    """``shasum -a 256`` output -> {filename: sha256}, bare names only."""
    sums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = DIGEST_RE.match(line)
        if not match:
            raise SystemExit(f"{path}: cannot read the checksum line {line!r}")
        name = match.group("name")
        if "/" in name:
            raise SystemExit(f"{path}: {name!r} is a path, not a bare filename")
        sums[name] = match.group("sha").lower()
    if not sums:
        raise SystemExit(f"{path} is empty; nothing was published")
    return sums


def build(dmg: Path, sums: dict[str, str], *, version: str, tag: str, repo: str,
          released_at: str | None = None) -> dict:
    name = dmg.name
    if name not in sums:
        raise SystemExit(f"{name} is not in SHA256SUMS ({', '.join(sorted(sums))})")
    payload = {
        "version": version,
        "url": f"https://github.com/{repo}/releases/download/{tag}/{name}",
        "filename": name,
        "sha256": sums[name],
        "size": dmg.stat().st_size,
        "notes_url": f"https://github.com/{repo}/releases/tag/{tag}",
        "released_at": released_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return validate_manifest(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="app version, e.g. 0.9.0")
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.9.0")
    parser.add_argument("--repo", default="Salcedoandrew/cullr", help="owner/name")
    parser.add_argument("--sums", type=Path, required=True, help="SHA256SUMS to read")
    parser.add_argument("--dmg", type=Path, required=True, help="the .dmg that was packed")
    parser.add_argument("--out", type=Path, required=True, help="where to write update.json")
    parser.add_argument("--current", help="the previous version, for the sanity check")
    args = parser.parse_args(argv)

    if not args.dmg.is_file():
        raise SystemExit(f"no such .dmg: {args.dmg}")
    try:
        payload = build(args.dmg, read_sums(args.sums), version=args.version,
                        tag=args.tag, repo=args.repo)
    except UpdateError as error:
        raise SystemExit(f"the app would refuse this manifest: {error}") from None

    # A manifest that is not newer than what is already out would silently do
    # nothing for every user clicking Check for updates, so say so here.
    if args.current and not is_newer(payload["version"], args.current):
        raise SystemExit(f"{payload['version']} is not newer than {args.current}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out} ({payload['filename']}, {payload['size']:,} bytes)")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
