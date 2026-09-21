"""Create an allowlisted code bundle; never include keys, datasets or predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = sorted(set([root / "pyproject.toml", root / "README.md"]
        + list((root / "iaa_agent").glob("*.py"))
        + list((root / "scripts").glob("*.py")) + list((root / "scripts").glob("*.sh"))
        + list((root / "tests").glob("*.py"))
        + list((root / "docs").glob("AUTONOMOUS_AGENT*.md"))))
    if args.out.exists():
        raise FileExistsError("Choose a new bundle path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(f.relative_to(root)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with tarfile.open(args.out, "w:gz") as archive:
        for f in files:
            archive.add(f, arcname=str(f.relative_to(root)).replace("\\", "/"), recursive=False)
    info = {"files": manifest, "archive_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "bytes": args.out.stat().st_size, "excludes": ["API keys", "datasets", "outputs", "images"]}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "files"}))


if __name__ == "__main__":
    main()
