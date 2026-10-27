#!/usr/bin/env python3
"""Render the GitHub release notes from README.md.

The feature catalogue is written once, in the ``## ✨ What it does`` section of
``README.md`` between the ``features:start`` / ``features:end`` markers, and the
limitations once between ``limits:start`` / ``limits:end``. The release workflow
runs this script and publishes its stdout, so a release can neither advertise a
feature the README does not describe nor miss one it does.

    python3 scripts/release_notes.py --check
    python3 scripts/release_notes.py --version 0.2.0 --previous v0.1.0

Standard library only: it runs before/independently of the project venv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SECTIONS = ("features", "limits")

INTRO = """\
Local-first AI photo culling for macOS. Point it at a folder, get back the keepers:
sharp frames, open eyes, and one best shot from every burst — plus a guarantee that
every person in the shoot appears in at least one keeper.

Everything runs on your machine: no uploads, no account, no cloud processing.
"""

INSTALL = """\
## Install

1. Download **`Cullr-{version}-arm64.dmg`** below (Apple silicon).
2. Open it and drag **Cullr** into Applications.
3. Right-click the app → **Open** the first time. The build is ad-hoc signed, not
   notarised, so Gatekeeper blocks a plain double-click once.

The UI, the face model and the Python runtime are all inside the bundle — nothing
else to install. Catalog, preview cache and logs live in
`~/Library/Application Support/Cullr`. Verify the download against `SHA256SUMS`;
dependency and model licences are in `NOTICE.md`, also inside the app bundle.
"""


def section(text: str, name: str) -> str:
    """Return the README content between ``<!-- name:start -->`` and its end."""
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    if start not in text or end not in text:
        raise SystemExit(
            f"README.md is missing the {start} / {end} markers; the release notes "
            f"are generated from that block."
        )
    body = text.split(start, 1)[1].split(end, 1)[0].strip("\n")
    if not body.strip():
        raise SystemExit(f"README.md has an empty {name} block.")
    return body


def render(version: str, tag: str, repo: str, previous: str | None = None) -> str:
    readme = README.read_text(encoding="utf-8")
    parts = [
        INTRO.strip("\n"),
        INSTALL.format(version=version).strip("\n"),
        "## What's in this build",
        section(readme, "features"),
        "## Not in this build",
        section(readme, "limits"),
    ]
    if previous:
        parts.append(
            f"**Full Changelog**: https://github.com/{repo}/compare/{previous}...{tag}"
        )
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="app version, e.g. 0.2.0")
    parser.add_argument("--tag", help="tag shown in the changelog link (default v<version>)")
    parser.add_argument("--repo", default="Salcedoandrew/cullr", help="owner/name")
    parser.add_argument("--previous", help="previous tag, for the compare link")
    parser.add_argument("--out", help="write here instead of stdout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="only validate that the README markers exist and are non-empty",
    )
    args = parser.parse_args(argv)

    readme = README.read_text(encoding="utf-8")
    for name in SECTIONS:
        section(readme, name)

    if args.check:
        print("README.md feature and limit blocks are present.")
        return 0

    if not args.version:
        parser.error("--version is required unless --check is used")

    notes = render(
        version=args.version,
        tag=args.tag or f"v{args.version}",
        repo=args.repo,
        previous=args.previous,
    )
    if args.out:
        Path(args.out).write_text(notes, encoding="utf-8")
        print(f"wrote {args.out} ({len(notes)} chars)")
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
