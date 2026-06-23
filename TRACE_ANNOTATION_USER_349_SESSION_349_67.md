# Agent Trace 注释：user 349 / session 349_67

## 文件来源

- Trace 文件：`outputs/eval_runs/user_349_deepseek_v2/user_349_session_349_67.json`
- 评估命令：

```powershell
python -m iaa_agent evaluate --user-id 349 --llm deepseek --save-runs outputs/eval_runs/user_349_deepseek_v2 --out outputs/evaluation/user_349_deepseek_v2.json
```

## 本次预测任务

- 用户：`349`
- Session：`349_67`
- 目标时间：`2013-02-14 07:25:39`
- Ground truth：`P004800`
- Ground truth 类别：`Deli / Bodega`
- 最后一个已知 POI：`P004082`
- 最后一个已知类别：`Deli / Bodega`
- 距离下一次签到的时间间隔：约 `0.27` 分钟
- 结果：ground truth 排名第 `2`，因此 Hit@5 和 Hit@10 命中。

本例的直观含义是：用户刚刚在一个 `Deli / Bodega` 签到，十几秒后又去了另一个非常近的 `Deli / Bodega`。agent 需要判断这更像是一次早晨固定路线中的短距离连续访问，而不是一次新的长距离探索。

## Agent 每一步在做什么

### 1. ObserveContext

trace 记录：

```text
4 visible check-ins
target hour 7
last category Deli / Bodega
```

这一步把当前 session 中 target 之前的签到整理成短期上下文。本例短期路径是：

```text
Gas Station / Garage -> Asian Restaurant -> School -> Deli / Bodega
```

agent 还会记录最后位置、目标小时、星期、最近类别序列、最近移动距离等结构化信息。这个阶段不做推荐，只负责把“当前发生了什么”整理成可供后续模块使用的状态。

### 2. BuildUserProfile

trace 记录：

```text
580 visible historical check-ins
top categories: School, Residential Building (Apartment / Condo), Deli / Bodega
```

这一步构造用户长期画像。它只使用用户前 80% 长期历史和当前 target 之前可见的 session context，不使用 target 本身。

本例中，用户 349 的高频类别包括：

```text
School
Residential Building (Apartment / Condo)
Deli / Bodega
```

这说明 `Deli / Bodega` 不是孤立类别，而是用户长期行为中的重要类别之一。

### 3. FindPeerUsers

trace 记录：

```text
5 peers selected
```

这一步根据类别-时间分布和粗粒度空间重合寻找相似用户。peer evidence 在稀疏用户上更重要；对用户 349 这种历史较丰富的用户，它更多是辅助信号。

### 4. InferIntention

DeepSeek 输出的意图摘要：

```text
User is in a morning routine near a school and deli, likely continuing daily pattern.
```

结构化 likely categories：

```text
Deli / Bodega: 0.6
School: 0.3
Food & Drink Shop: 0.1
```

这一步的作用是把上下文和长期画像压缩成一个可解释的意图对象。本例中，LLM 判断用户处于早晨 routine，且最近路径中的 `School` 和 `Deli / Bodega` 与用户长期习惯一致。

缓存记录：

```text
DeepSeek cache hit/miss tokens: 1152/1277
```

这说明 DeepSeek 的输入前缀缓存已经命中了一部分 prompt。命中的主要是固定任务说明和同一用户的稳定长期画像。

### 5. PlanTools

trace 记录：

```text
5 tools planned
```

agent 规划了以下召回工具：

```text
HistoricalRecall
SpatialRecall
CategoryIntentRecall
TransitionRecall
PeerRecall
```

这些工具分别从用户历史、当前位置附近、意图类别、转移规律和相似用户中召回候选 POI。

### 6. Candidate Retrieval

第一轮召回：

```text
HistoricalRecall: 60 records/candidates observed
SpatialRecall: 50 records/candidates observed
CategoryIntentRecall: 1359 records/candidates observed
TransitionRecall: 30 records/candidates observed
PeerRecall: 32 records/candidates observed
FilterCandidates: raw=203, filtered=30
```

这里的 `CategoryIntentRecall` 很大，是因为它会在 POI universe 中查找与 `Deli / Bodega`、`School`、`Food & Drink Shop` 匹配的 POI。之后 `FilterCandidates` 会合并去重，并保留 prior score 较高的一批候选。

### 7. Reflection

第一轮排序后触发了 reflection：

```text
top_scores_close
expand_spatial_radius
add_category_transition_peer_candidates
rerank_with_affordances
```

含义是：top 候选之间分差太小，agent 认为当前候选池不够稳定，于是扩大空间召回、补充类别转移和 peer 候选，再重新计算 affordance。

第二轮候选池：

```text
raw=361
filtered=60
```

这让最终排序有更大的候选覆盖面。

### 8. Final Ranking

最终 top-5：

| rank | poi_idx | category | alignment_score | 说明 |
|---:|---|---|---:|---|
| 1 | P004082 | Deli / Bodega | 0.8135 | 最后一个已知 POI，本身高度匹配 routine |
| 2 | P004800 | Deli / Bodega | 0.8135 | ground truth，距离约 0.01 km |
| 3 | P002766 | School | 0.7904 | 用户长期高频类别，且距离很近 |
| 4 | P003803 | Food & Drink Shop | 0.7585 | 意图相关类别，距离很近 |
| 5 | P004014 | School | 0.6493 | 类别匹配，但距离略远 |

ground truth `P004800` 排第 2。它被推荐的主要原因是：

- 类别 `Deli / Bodega` 是 DeepSeek intention 的最高权重类别。
- 它距离最后位置约 `0.01 km`，空间可达性非常强。
- 用户长期画像中 `Deli / Bodega` 是重要类别。
- 当前 target 距离上一条签到只有十几秒，短距离连续访问更合理。

## 为什么不是 rank 1

rank 1 是 `P004082`，也就是最后一个已知 POI。它与 ground truth 类别相同、距离为 0、历史证据也强，因此 affordance 分数与 ground truth 几乎相同。

这暴露了一个后续可以优化的问题：当 candidate 等于 last-known POI 时，除非用户有强重访证据，否则应该轻微惩罚“原地不动”的候选。当前系统允许它进入 top1，所以 ground truth 被挤到了 rank 2。

## 这个 case 说明了什么

这个 trace 展示了系统的完整工作流已经形成：

```text
observe context
-> build user profile
-> infer intention with DeepSeek
-> plan recall tools
-> retrieve candidates
-> compute affordances
-> reflect and expand
-> rerank
-> explain final ranking
```

它也说明 session-level eval 比 event-level 滑窗更适合本项目：短期路径在同一个 trajectory 内，意图解释自然，而且每个 session 只产生一个预测目标。

