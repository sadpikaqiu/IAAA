# IAA-Agent NYC-first

CLI + JSON demo for an Intention-Affordance Aligned Agent for next POI recommendation.

The current v0 targets the local Foursquare NYC split under `datasets/NYC` and implements a structured mobility affordance workflow:

```text
observe context -> infer intention -> plan tools -> retrieve candidates
-> build affordance profiles -> align -> reflect -> rank/explain
```

## Data

Data files are intentionally not committed. Place the following files locally:

```text
datasets/NYC/NYC_train.csv
datasets/NYC/NYC_val.csv
datasets/NYC/NYC_test.csv
```

The v0 data boundary excludes reviews, images, opening hours, price, and ratings. The agent records these as missing evidence instead of hallucinating unsupported claims.

## Install

```powershell
python -m pip install -e .
```

## Usage

```powershell
python -m iaa_agent prepare --data-dir datasets/NYC
python -m iaa_agent run --traj-id 349_52 --out outputs/runs/smoke_349_52.json
python -m iaa_agent replay --case cases/case_a.json
python -m iaa_agent evaluate --limit 50
```

The default LLM mode is deterministic `fake`, so tests and normal smoke runs do not require network access.

To use DeepSeek:

```powershell
$env:DEEPSEEK_API_KEY = "<your key>"
python -m iaa_agent run --traj-id 349_52 --llm deepseek
```

Never commit API keys or `.env` files.

## Test

```powershell
python -m pytest -q
```

