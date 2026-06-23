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
python -m iaa_agent user-targets --user-id 349
python -m iaa_agent run-user --user-id 349 --out outputs/runs/user_349_tail.json
python -m iaa_agent run-user --user-id 349 --target-index 576 --out outputs/runs/user_349_576.json
python -m iaa_agent replay --case cases/case_a.json
python -m iaa_agent evaluate --user-id 349 --out outputs/evaluation/user_349_session_results.json
python -m iaa_agent evaluate --user-id 349 --save-runs outputs/eval_runs/user_349
python -m iaa_agent evaluate
python -m iaa_agent evaluate --smoke-limit 50
```

The default LLM mode is deterministic `fake`, so tests and normal smoke runs do not require network access.

`run --traj-id` is kept for GETNext-style trajectory debugging. `run-user` is for inspecting one event-level case and its full agent trace.

For `run-user`, `--target-index` is optional. If omitted, the CLI predicts the last held-out event for that user. Use `user-targets` first when you want to inspect the valid index range and choose a specific test point.

`evaluate` is the formal session-level evaluation path. It sorts each user's full check-in stream, uses the first 80% as long-term history, then evaluates original `trajectory_id` sessions whose final check-in falls in the held-out 20%. Each session contributes one prediction: previous check-ins in that trajectory are the short-term context, and the final check-in is the ground truth.

Recommended evaluation workflow:

- Unit logic tests: `python -m pytest -q`
- Single-user evaluation: `python -m iaa_agent evaluate --user-id 349`
- Single-user traces: `python -m iaa_agent evaluate --user-id 349 --save-runs outputs/eval_runs/user_349`
- Full evaluation: `python -m iaa_agent evaluate`

Use `--save-runs` when you need per-session `AgentRunResult` JSON files for case study and error analysis. Use `--smoke-limit` only for quick development runs; omit it for full-dataset reporting.

Outputs expose both IDs:

- `poi_idx`: stable compact ID such as `P000123`, intended for prompts and readable traces.
- `poi_id`: original Foursquare ID, retained for evaluation and data provenance.

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
