# IAA-Agent NYC-first Implementation Plan

## Objective

Implement a CLI + JSON v0 demo for next POI recommendation on `datasets/NYC`.
The system follows the full agentic workflow:

```text
observe context -> infer intention -> plan tools -> retrieve candidates
-> build structured mobility affordances -> align -> reflect -> rank/explain
```

## Data Boundary

The NYC split supports category, coordinates, timestamps, weekday, normalized
time, and trajectory ids. It does not support reviews, images, opening hours,
price, ratings, crowding, or social atmosphere. These absent fields must appear
as `missing_evidence` and must not be hallucinated in explanations.

## Interfaces

- `iaa-agent prepare --data-dir datasets/NYC`
- `iaa-agent run --traj-id <test_trajectory_id> --out outputs/runs/<id>.json`
- `iaa-agent user-targets --user-id <user_id>`
- `iaa-agent run-user --user-id <user_id> [--target-index <idx>] --out outputs/runs/<id>.json`
- `iaa-agent replay --case cases/case_a.json`
- `iaa-agent evaluate --limit 50`
- `iaa-agent evaluate-user-split --limit 50`

The default LLM mode is `fake`, which is deterministic and does not require
network access. Live DeepSeek calls are enabled only with `--llm deepseek` and
`DEEPSEEK_API_KEY` in the environment.

`run --traj-id` remains available for GETNext-style session debugging. The main
research/evaluation path is now user chronological splitting: sort each user's
full check-in stream by time, use the first 80% as long-term history, and
predict events in the remaining 20% with the preceding check-ins as short-term
context. If `run-user` omits `--target-index`, it defaults to the user's last
held-out event.

Every POI has two IDs in outputs:

- `poi_idx`: stable compact ID (`P000001`) for prompts and readable traces.
- `poi_id`: original Foursquare ID for provenance and evaluation.

## Acceptance Criteria

- A test trajectory produces a JSON `AgentRunResult` with context, user profile,
  inferred intention, tool plan, candidate source summary, top-10 ranked POIs,
  affordance profiles, reflection record, and trace.
- Every ranked POI includes score decomposition and at least three supporting
  evidence statements.
- Guardrails prevent unsupported claims about reviews, photos, opening hours,
  price, or ratings.
- Evaluation reports Hit@1/5/10, NDCG@1/5/10, and MRR.

## Future Fork

Yelp adaptation should be implemented as a separate richer-affordance fork.
It can fill the reserved dataset capability fields and add review, price,
rating, and ambience evidence without changing the v0 JSON contract.
