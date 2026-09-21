"""Run WWW2024 POI image summarization from a checkout without installing it."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iaa_agent.poi_image_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
