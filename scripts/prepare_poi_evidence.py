"""Freeze WWW2024 comments and completed visual summaries for evaluation."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iaa_agent.evidence import build_snapshot, write_snapshot
from iaa_agent.poi_image_summary import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=["NYC", "TKY", "both"], default="both")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--image-summaries", type=Path, default=REPO_ROOT / "outputs/poi_image_summaries")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/poi_evidence")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--unavailable-manifest", type=Path, help="Audited provider rejections retained as missing visual evidence")
    parser.add_argument("--allow-partial", action="store_true", help="Write development snapshot; evaluation will reject it")
    args = parser.parse_args(argv)
    snapshots = []
    try:
        for city in (["NYC", "TKY"] if args.city == "both" else [args.city]):
            snapshot = build_snapshot(args.data_root, args.image_summaries, city,
                                      verify_image_bytes=not args.check_only,
                                      unavailable_manifest=args.unavailable_manifest)
            snapshots.append(snapshot)
            coverage = dict(snapshot["coverage"])
            coverage["missing_image_pois"] = len(coverage["missing_image_pois"])
            coverage["invalid_image_pois"] = len(coverage["invalid_image_pois"])
            coverage["unavailable_image_pois"] = len(coverage["unavailable_image_pois"])
            coverage["summaries_with_unreadable_images"] = len(coverage["summaries_with_unreadable_images"])
            print(json.dumps({"city": city, **coverage}, ensure_ascii=False), flush=True)
        if args.check_only:
            # This quick check is not the byte-verified readiness certificate.
            return 0 if all(not s["coverage"]["missing_image_pois"] and
                            not s["coverage"]["invalid_image_pois"] and
                            not s["coverage"]["generation_active"] and
                            len(s["image_configurations"]) == 1 for s in snapshots) else 3
        if not args.allow_partial and any(not s["coverage"]["complete"] for s in snapshots):
            raise ValueError("Image generation is not complete/consistent for all requested cities")
        for snapshot in snapshots:
            path = args.output_dir / f"{snapshot['city']}_{snapshot['snapshot_id'][:12]}.json"
            write_snapshot(snapshot, path, allow_partial=args.allow_partial)
            print(f"Snapshot: {path}", flush=True)
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
