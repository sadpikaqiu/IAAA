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
python -m iaa_agent compare-p4 --concurrency 4
python -m iaa_agent evaluate --llm deepseek --variant p4v1 --model deepseek-v4-flash --concurrency 8 --stall-timeout 600 --report-stratified --out outputs/evaluation/p4v1_deepseek_full_ih_ooh.json
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

For IH/OOH evaluation, `IH` means that the target POI appears in the user's first 80% chronological long-term history; `OOH` means that it does not. The current session's short-term context does not change this label. In this single-ground-truth next-POI setting, `Acc@K` is numerically identical to `Hit@K`. `--report-stratified` computes both subsets from the same prediction ranks, so it does not trigger a second LLM pass.

DeepSeek evaluation is fault-tolerant by default. A valid model response without `usage` metadata is retained and counted in `usage_missing_count`, not `fallback_count`. If intention generation genuinely fails, the session uses the deterministic heuristic intention and is recorded in `llm_anomalies`; the remaining evaluation continues. Use `--no-allow-fallback` to mark such sessions as strict violations without aborting or discarding completed work.

DeepSeek cache notes are recorded in `docs/DEEPSEEK_CONTEXT_CACHE_NOTES.md`; an annotated successful trace is in `docs/TRACE_ANNOTATION_USER_349_SESSION_349_67.md`.

A Chinese conference-style project report for the current P4v1 mainline is available at `docs/PROJECT_REPORT_P4V1_CN.md`.

Outputs expose both IDs:

- `poi_idx`: stable compact ID such as `P000123`, intended for prompts and readable traces.
- `poi_id`: original Foursquare ID, retained for evaluation and data provenance.

To use DeepSeek:

```powershell
$env:DEEPSEEK_API_KEY = "<your key>"
python -m iaa_agent run --traj-id 349_52 --llm deepseek
```

Never commit API keys or `.env` files.

To serve the local `Qwen3.8-27B-FP8` checkpoint with vLLM on Linux, follow
`docs/QWEN38_VLLM_SERVER_SETUP.md`. The evaluation command uses `--llm openai`
and the standard `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`
environment variables.

## Test

```powershell
python -m pytest -q
```
