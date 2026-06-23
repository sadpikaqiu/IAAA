# IAA-Agent 数据路线决策：NYC-first

## 决策

当前阶段先使用 `datasets/NYC` 下已按访问时间排序并按 8:1:1 切分的 Foursquare NYC 数据，实现完整 IAA-Agent 系统。Context Trails + Foursquare API 补全暂不作为前置任务。

原因：

- 补全 Context Trails 的 Foursquare POI 属性存在 API credential、endpoint 兼容和配额成本问题。
- 当前优先级是验证完整 agent workflow，而不是一次性补齐 Agent4POI 级别的多模态/评论/营业时间 affordance。
- `datasets/NYC` 已包含轨迹、类别、坐标、本地时间、星期几和 trajectory id，足够支撑 v0 的 structured mobility affordance。

## 当前 NYC 数据能力

字段：

```text
user_id
POI_id
POI_catid
POI_catid_code
POI_catname
latitude
longitude
timezone
UTC_time
local_time
day_of_week
norm_in_day_time
trajectory_id
norm_day_shift
norm_relative_time
```

本地统计：

| split | check-ins | users | POIs | categories | trajectories |
|---|---:|---:|---:|---:|---:|
| train | 83,228 | 1,047 | 4,980 | 207 | 11,022 |
| val | 10,339 | 513 | 2,558 | 180 | 1,559 |
| test | 10,374 | 461 | 2,403 | 182 | 1,566 |

额外检查：

- 总记录数 103,941，split 比例约为 0.801 / 0.099 / 0.100。
- test 中 2,375 / 2,403 个 POI 在 train+val 中出现。
- test 中 457 / 461 个用户在 train+val 中出现。
- test 中 182 / 182 个类别在 train+val 中出现。

## v0 Affordance 定位

v0 明确做 **structured mobility affordance**，不是 Agent4POI 的 multimodal affordance。

支持的 affordance：

- `category_match`：候选 POI 类别是否符合 inferred intention。
- `spatial_feasibility`：候选 POI 与最后已知 POI 的距离是否在用户典型移动范围内。
- `temporal_fit`：候选 POI 或候选类别是否常在目标小时/星期被访问。
- `revisit_support`：用户是否访问过该 POI 或同类 POI。
- `transition_support`：从最近 POI/category 到候选 POI/category 的历史转移是否常见。
- `peer_support`：相似用户在目标时间窗口是否访问过该 POI 或同类 POI。
- `popularity_support`：候选 POI/category 在目标时间窗口是否具有全局热门性。
- `reachability_time_gap`：结合 query trajectory 最后一跳时间差和距离，判断移动是否合理。

不支持的 affordance：

- 评论语义、图片场景、真实营业时间、价格、评分、拥挤程度、社交氛围。
- 系统必须把这些写入 `missing_evidence`，不能生成相关事实断言。

## 实施影响

- 完整 agent 系统仍然成立：observe context -> infer intention -> plan tools -> retrieve candidates -> build affordance profiles -> align -> reflect -> rank/explain。
- 论文叙述上要诚实区分：本项目 v0 是 IntentPOI + structured affordance agent；Agent4POI 提供的是 item-side context-conditioned affordance 思想，不是当前数据层的完全复现。
- 后期可 fork 一个 Yelp 版本作为 full affordance 版本，用评论、评分、价格等字段增强 item-side evidence。

## 下一步

1. 将项目计划从 “先补全 Context Trails” 调整为 “NYC-first 完整系统优先”。
2. 数据加载器直接适配 `datasets/NYC/NYC_train.csv`、`NYC_val.csv`、`NYC_test.csv`。
3. 建立 dataset capability guardrail，明确 `has_reviews=false`、`has_images=false`、`has_opening_hours=false`、`has_price=false`、`has_ratings=false`。
4. 先实现可离线验证的 heuristic tools，再接入 DeepSeek 做 intention 和 explanation。

