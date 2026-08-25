# IAA-Agent 需求文档

版本：v0.1  
项目名：IAA-Agent / IAR-POI  
全称：Intention-Affordance Aligned Agent for Next POI Recommendation  
目标阶段：工程 Demo 需求文档  
当前重点：先实现一个不用图像、不用评论、基于结构化 POI 轨迹数据的小步快跑 Demo

---

## 0. 项目一句话

构建一个面向 next POI recommendation 的 agent demo：系统先观察用户轨迹与时间空间上下文，推断用户当前 intention，再主动调用候选召回与证据工具，构建候选 POI 的 affordance profile，最后完成 **intention-affordance alignment** 排序与解释。

核心不是让 LLM 一次性回答“下一个 POI 是什么”，而是：

```text
Observe context
→ infer intention
→ plan tools
→ retrieve candidates
→ inspect affordances
→ align intention and affordance
→ reflect if evidence insufficient
→ rank and explain
```

这个设计吸收了 IntentPOI 的 “thinking before acting” 思路，也吸收了 Agent4POI 的 “POI 在情境下提供 affordance” 思路，但我们进一步把二者合并到一个 tool-augmented agent 框架里。

---

## 1. Demo 目标与非目标

### 1.1 Demo 目标

v0 demo 需要完成以下能力：

1. 给定一个用户的历史签到轨迹、当前 query trajectory、目标预测时间，系统能输出 Top-10 推荐 POI。
2. 系统必须显式生成用户当前 intention，而不是直接预测 POI。
3. 系统必须通过工具构造候选池，而不是让 LLM 从全量 POI 中自由生成。
4. 系统必须为候选 POI 构建 affordance profile，说明每个候选为什么能或不能满足 intention。
5. 系统必须输出推荐理由、证据来源、缺失证据和不确定性。
6. 系统必须保留完整 agent trace，展示每一步调用了哪些工具、为什么调用、观察到了什么。

### 1.2 v0 非目标

v0 暂时不做：

- 图像理解；
- 外部 Google Places / Yelp API 补全；
- 真实线上部署；
- 训练模型；
- 大规模 benchmark；
- 复杂多 agent 协作；
- 完整论文实验对比；
- 对所有 POI 做昂贵证据检索。

v0 的目标是先把 **agentic intention-affordance workflow** 跑通。

---

## 2. 数据集选择：v0 先不用图像，也尽量先不用评论

### 2.1 推荐 v0 数据源

v0 建议优先使用标准 next POI 数据：

```text
Foursquare NYC
Foursquare TKY
可选：CA / Gowalla-like dataset
```

优先字段：

```text
user_id
poi_id
poi_name
poi_category
timestamp
latitude
longitude
trajectory/session_id，如果有
```

这类结构化 POI 轨迹数据已经足够支撑 v0 的 intention inference、candidate retrieval 和 structured affordance construction。

### 2.2 为什么 v0 不建议先上 Yelp 评论？

Yelp 自带评论，确实适合做 affordance evidence，比如“适合约会”“安静”“服务慢”“适合聚会”。但问题是：

1. Yelp 不是大多数 next POI prediction 论文的主流标准数据，后续做对比实验会多一层数据适配工作。
2. 评论处理会引入额外复杂度：评论时间过滤、review summarization、噪声、token 成本。
3. 如果一开始就上评论，很容易把工程重点变成“评论理解”，而不是验证 agent workflow。
4. v0 的 affordance 其实可以先从结构化轨迹和 POI metadata 中构建。

所以 v0 先不依赖评论。评论作为 v1 插件。

### 2.3 v0 没有图像/评论时，affordance 从哪里来？

v0 的 affordance 不做“环境氛围”这种视觉/评论型判断，而做结构化 affordance：

```text
category_affordance       候选 POI 类别是否满足 intention
temporal_affordance       目标时间是否符合用户/全局访问模式
spatial_affordance        候选 POI 是否在用户可接受移动范围内
revisit_affordance        用户是否有重访该 POI 或同类 POI 的倾向
transition_affordance     从最近 POI/category 转移到该候选是否合理
peer_affordance           相似用户在类似时间是否访问该 POI/category
popularity_affordance     该 POI/category 在目标时间是否常被访问
exploration_affordance    如果用户有探索倾向，未访问 POI 是否仍合理
```

这足够支撑一个 demo。v1 再扩展：

```text
review_affordance
opening_hours_affordance
price_affordance
crowd_affordance
photo_affordance
```

---

## 3. 核心概念定义

### 3.1 Intention：用户当前想做什么

Intention 是用户侧需求结构，不只是自然语言 summary。

v0 intention schema：

```json
{
  "activity_goal": "evening food-and-drink / social outing",
  "likely_categories": [
    {"category": "restaurant", "weight": 0.32},
    {"category": "wine bar", "weight": 0.25},
    {"category": "gastropub", "weight": 0.20},
    {"category": "dessert shop", "weight": 0.12},
    {"category": "nightlife spot", "weight": 0.11}
  ],
  "temporal_requirements": [
    "target time is Wednesday 21:00",
    "night-time venue is preferred"
  ],
  "spatial_requirements": {
    "anchor_location": "last known POI",
    "preferred_radius_km": 2.0,
    "allow_long_jump": false
  },
  "behavioral_requirements": {
    "revisit_tendency": "medium-high",
    "exploration_tendency": "medium",
    "peer_influence": "medium"
  },
  "evidence": [
    "user often visits bars and restaurants after work",
    "recent trajectory ends at cinema",
    "similar users around 21:00 visit food-and-drink venues"
  ],
  "confidence": 0.78,
  "uncertainty_reasons": [
    "query trajectory is short",
    "multiple categories plausible"
  ]
}
```

### 3.2 Affordance：候选 POI 当前能提供什么

Affordance 是候选 POI 在当前 context 下满足用户 intention 的能力。

v0 affordance schema：

```json
{
  "poi_id": "p_123",
  "poi_name": "Barcibo Enoteca",
  "category": "wine bar",
  "affordances": [
    {
      "name": "category_match",
      "requirement": "food-and-drink / wine bar",
      "answer": "yes",
      "confidence": 0.92,
      "evidence": "POI category is wine bar"
    },
    {
      "name": "spatial_feasibility",
      "requirement": "near last known location",
      "answer": "yes",
      "confidence": 0.84,
      "evidence": "distance from last POI is 0.6 km"
    },
    {
      "name": "temporal_fit",
      "requirement": "night-time visit pattern",
      "answer": "yes",
      "confidence": 0.76,
      "evidence": "user has visited similar categories around 20:00-22:00"
    },
    {
      "name": "revisit_support",
      "requirement": "known or same-style place",
      "answer": "uncertain",
      "confidence": 0.48,
      "evidence": "user has not visited this POI but has visited similar wine bars"
    }
  ],
  "missing_evidence": [
    "opening hours unavailable",
    "review sentiment unavailable"
  ],
  "conflicts": [],
  "alignment_score": 0.83
}
```

### 3.3 Alignment：用户意图与 POI affordance 的对齐

最终排序不是简单 popularity，也不是 LLM 直接打分，而是：

```text
用户当前想做什么
×
候选地点能否支持这件事
```

形式上：

```text
Score(u,p,c) = Align(I(u,c), A(p,c))
```

v0 可以先做加权 alignment：

```text
alignment_score =
  w_category   * category_match
+ w_temporal   * temporal_fit
+ w_spatial    * spatial_feasibility
+ w_revisit    * revisit_support
+ w_transition * transition_support
+ w_peer       * peer_support
+ w_popularity * temporal_popularity
+ w_explore    * exploration_fit
```

权重由 intention 动态决定。例如：

- intention 很明确是“late-night food-and-drink”，category 和 temporal 权重更高；
- 用户强重访倾向明显，revisit 权重更高；
- 用户是 sparse user，peer 和 global temporal popularity 权重更高；
- 最近位置很重要，spatial 权重更高。

---

## 4. 整体系统流程

### 4.1 高层流程

```text
Input:
  user u
  historical check-ins H_u
  query trajectory X_q
  target timestamp t_q
  POI catalog D

Step 1: Observe Context
Step 2: Build User Profile
Step 3: Infer Intention
Step 4: Plan Candidate Retrieval Tools
Step 5: Retrieve Candidate Pool
Step 6: Filter Candidate Pool
Step 7: Build Lightweight Affordance Profiles
Step 8: Selectively Deepen Evidence
Step 9: Align Intention-Affordance
Step 10: Reflect if Evidence Insufficient
Step 11: Final Ranking + Explanation
```

### 4.2 Agent 状态机

```text
S0_CONTEXT_OBSERVED
S1_INTENTION_INFERRED
S2_TOOL_PLAN_READY
S3_CANDIDATES_RETRIEVED
S4_CANDIDATES_FILTERED
S5_AFFORDANCES_BUILT
S6_ALIGNMENT_SCORED
S7_REFLECTION_DONE
S8_FINAL_OUTPUT
```

每个状态都必须记录：

```json
{
  "state": "S3_CANDIDATES_RETRIEVED",
  "agent_thought_summary": "Need both revisit and nearby candidates because user has medium-high repeat tendency but last location changed.",
  "tools_called": ["HistoricalRecall", "SpatialRecall", "CategoryIntentRecall"],
  "observations": [],
  "next_action": "Filter candidates to B=30"
}
```

注意：这里记录的是可公开的简短 reasoning summary，不需要保存完整隐藏 chain-of-thought。

---

## 5. Observe Context 阶段需求

Observe Context 是整个 agent 的输入整理阶段。它不做推荐，只把后续推理需要的信息整理成 **Context Snapshot**。

### 5.1 必须纳入的 context

#### A. Query context

```text
target_timestamp
target_hour
target_day_of_week
is_weekend
time_of_day_bucket: morning / noon / afternoon / evening / night / late_night
query_trajectory
last_known_poi
last_known_category
last_known_location
time_gap_since_last_checkin
```

#### B. Short-term trajectory context

取 query trajectory 最近 K 个 check-ins，默认 K=5。

需要生成：

```text
recent_poi_sequence
recent_category_sequence
recent_time_sequence
recent_spatial_movement
recent_transition_pattern
session_stage: start / middle / after_event / late_session
```

例子：

```text
restaurant → bar → cinema
target time = 21:00
可能意图：after-cinema food/drink/social venue
```

#### C. Long-term user context

从历史轨迹生成用户画像，至少包括：

```text
top visited POIs
top categories
hourly visit distribution
day-of-week distribution
category-by-hour matrix
category-by-day matrix
revisit ratio
unique POI ratio
average movement distance
max common movement distance
home/activity clusters, if 可计算
frequent transitions: category A → category B
frequent same-time POIs/categories
```

#### D. Spatial context

```text
last_location
distance from last POI to candidate POIs
user typical movement radius
city-level POI density around last location
geohash cell of current region
```

#### E. Peer context，v0.5 可选，v0 可以先做简化

如果工程时间允许，加入 peer behavior。

v0 可以先做简单版本：

```text
similarity = category-time profile similarity + geohash overlap
peer window = target time ± 30 minutes
top_k_peers = 5
```

输出：

```text
peer_top_categories_at_target_time
peer_top_pois_near_region
peer_common_transitions
```

#### F. Dataset capability context

系统必须知道当前数据集有什么、没有什么，避免 hallucination。

```json
{
  "has_reviews": false,
  "has_images": false,
  "has_opening_hours": false,
  "has_price": false,
  "has_categories": true,
  "has_coordinates": true,
  "has_poi_names": true,
  "has_timestamps": true
}
```

如果没有 opening hours，agent 不能说“该店现在营业”。只能说：

```text
“该类别/POI 在历史数据中常于该时间被访问”
```

而不能说：

```text
“该店 21:00 仍营业”
```

---

## 6. Intention Inference 需求

### 6.1 输入

```json
{
  "context_snapshot": {},
  "user_profile_summary": {},
  "short_term_trajectory_summary": {},
  "peer_behavior_summary": {},
  "dataset_capabilities": {}
}
```

### 6.2 输出

必须输出结构化 JSON：

```json
{
  "summary": "The user is likely looking for an evening food-and-drink venue near the last known location.",
  "activity_goal": "evening food-and-drink",
  "likely_categories": [
    {"category": "restaurant", "weight": 0.35, "evidence": "frequent evening visits"},
    {"category": "bar", "weight": 0.25, "evidence": "recent trajectory includes nightlife pattern"},
    {"category": "dessert shop", "weight": 0.15, "evidence": "post-dinner pattern"}
  ],
  "spatial_preference": {
    "anchor": "last_known_location",
    "preferred_radius_km": 2.0,
    "allow_long_distance": false,
    "evidence": "user's median transition distance is 1.3 km"
  },
  "temporal_preference": {
    "target_hour": 21,
    "preferred_time_bucket": "night",
    "evidence": "user often visits food/drink POIs from 20:00 to 22:00"
  },
  "behavioral_preference": {
    "revisit_tendency": 0.72,
    "exploration_tendency": 0.28,
    "peer_dependency": 0.40
  },
  "confidence": 0.78,
  "uncertainty_reasons": [
    "recent trajectory is short",
    "restaurant and bar are both plausible"
  ]
}
```

### 6.3 Intention 由哪些信息构建？

v0 不用评论、不用图像，intention 主要由这些信息构建：

```text
1. 用户长期访问类别分布
2. 用户目标小时段常访问类别
3. 用户目标星期常访问类别
4. 用户最近 K 个 POI/category 的序列模式
5. 最近 POI 到目标时间的时间间隔
6. 用户历史 transition pattern
7. 用户 revisit/explore 倾向
8. 当前位置和用户常见活动区域
9. 相似用户在目标时间附近的访问类别，可选
10. 全局同时间热门类别，可选
```

评论不是 v0 的必要条件。等 v1 加评论时，评论主要用于 **affordance evidence**，而不是 intention inference 的核心来源。

---

## 7. Tool Planning 阶段需求

Tool Planning 的目标是：让 agent 根据 intention 决定该调用哪些工具，而不是固定所有工具全跑。

### 7.1 Tool registry

#### A. Context tools

| Tool | 作用 | v0 是否需要 |
|---|---|---|
| BuildUserProfile | 生成长期用户画像 | 必须 |
| SummarizeQueryTrajectory | 总结短期轨迹 | 必须 |
| ComputeUserMobilityStats | 计算访问频率、时间 pattern、距离 pattern | 必须 |
| FindPeerUsers | 找相似用户 | 可选 |
| GetPeerBehavior | 获取相似用户目标时间附近行为 | 可选 |

#### B. Candidate retrieval tools

| Tool | 作用 | v0 是否需要 |
|---|---|---|
| HistoricalRecall | 召回用户历史访问 POI | 必须 |
| SpatialRecall | 召回最近位置附近 POI | 必须 |
| CategoryIntentRecall | 根据 intention category 召回 POI | 必须 |
| TransitionRecall | 根据最近 POI/category 的历史转移召回 | 强烈建议 |
| PeerRecall | 根据相似用户访问召回 | 可选 |
| TemporalPopularityRecall | 根据目标时间全局热门 POI/category 召回 | 可选 |

#### C. Affordance tools

| Tool | 作用 | v0 是否需要 |
|---|---|---|
| CheckCategoryMatch | 检查类别是否满足 intention | 必须 |
| CheckSpatialFeasibility | 检查距离是否合理 | 必须 |
| CheckTemporalFit | 检查目标时间是否符合用户/全局访问模式 | 必须 |
| CheckRevisitSupport | 检查用户是否访问过该 POI 或同类 POI | 必须 |
| CheckTransitionSupport | 检查从最近 POI/category 到候选是否常见 | 强烈建议 |
| CheckPeerSupport | 检查 peer 是否支持 | 可选 |
| CheckExplorationFit | 检查未访问 POI 是否符合用户探索倾向 | 建议 |
| CheckOpeningHours | 检查营业时间 | v1 |
| SearchReviews | 检索评论证据 | v1 |
| SummarizeReviewAffordance | 评论转 affordance | v1 |
| GetPhotoCaptions | 图片证据 | v2 |

#### D. Reflection tools

| Tool | 作用 | v0 是否需要 |
|---|---|---|
| DetectEvidenceGap | 判断是否缺关键证据 | 必须 |
| DetectEvidenceConflict | 判断证据是否冲突 | 必须 |
| CompareTopCandidates | 对 top candidates 做 listwise 比较 | 必须 |
| ExpandCandidatePool | 候选不足时扩召回 | 必须 |
| RecalibrateConfidence | 根据缺失证据和冲突调整置信度 | 建议 |

---

## 8. Candidate Retrieval 设计

### 8.1 Agent 如何自己选候选集？

Agent 不直接从全量 POI 中“想象”候选，而是选择 recall tools 和参数。

例子：

```json
{
  "intention": "night food-and-drink",
  "tool_plan": [
    {
      "tool": "HistoricalRecall",
      "reason": "User has medium-high revisit tendency",
      "params": {"same_hour_boost": true, "same_day_boost": true}
    },
    {
      "tool": "SpatialRecall",
      "reason": "Last known location is available",
      "params": {"radius_km": 2.0, "limit": 50}
    },
    {
      "tool": "CategoryIntentRecall",
      "reason": "Likely categories are restaurant, wine bar, dessert shop",
      "params": {"categories": ["restaurant", "wine bar", "dessert shop"], "limit": 50}
    },
    {
      "tool": "TransitionRecall",
      "reason": "Recent trajectory ends at cinema, need after-cinema transitions",
      "params": {"last_category": "cinema", "limit": 30}
    }
  ]
}
```

### 8.2 初始候选池

建议 v0 设计：

```text
HistoricalRecall: top 50 或用户全部历史 POI
SpatialRecall: nearest 50
CategoryIntentRecall: top 50 within radius / city
TransitionRecall: top 30
PeerRecall: top 30，可选
TemporalPopularityRecall: top 30，可选
```

合并去重后得到 raw candidate pool：

```text
C_raw = union(all recall results)
```

通常 C_raw 可能是 50–150 个。

### 8.3 候选过滤

过滤到 LLM 能处理的规模：

```text
B = 30
```

v0 先固定 B=30。

过滤 prior score：

```text
candidate_prior =
  0.30 * historical_score
+ 0.20 * spatial_score
+ 0.20 * category_intent_score
+ 0.15 * transition_score
+ 0.10 * temporal_popularity_score
+ 0.05 * peer_score
```

这个 prior score 只用于进入候选池，不是最终排序分数。

### 8.4 候选池覆盖检查

过滤后需要检查：

```text
1. 是否覆盖 intention top categories？
2. 是否全部都是历史高频 POI？
3. 是否缺少空间附近 POI？
4. 是否缺少探索型 POI？
5. 是否候选过于单一？
```

如果出现以下情况，触发 ExpandCandidatePool：

```text
top intention category 没有候选
filtered candidates 少于 B
候选类别熵过低，过度集中在一个类别
所有候选距离都过远
所有候选都是历史 POI，没有 exploration candidate
```

---

## 9. Affordance Evidence Acquisition 设计

### 9.1 是全部候选都算，还是选择性算？

v0 采用 **两阶段策略**：

```text
Stage A: 对 B=30 个候选全部计算 lightweight affordances
Stage B: 只对 top M 或 evidence-gap 候选做 deeper evidence
```

因为 v0 没有评论和图像，所以 Stage A 成本很低，可以全算。未来 v1 加评论、营业时间、图片后，不能对所有候选都做昂贵证据检索，要选择性调用。

推荐配置：

```text
B = 30 进入 affordance scan
M = 10 进入 deep comparison
R = 5 进入 reflection
```

### 9.2 Lightweight affordance 维度

v0 对每个候选都计算：

```text
category_match
spatial_feasibility
temporal_fit
revisit_support
same_category_revisit_support
transition_support
peer_support，可选
temporal_popularity
exploration_fit
```

### 9.3 每个 affordance 的回答格式

```json
{
  "affordance_name": "transition_support",
  "requirement": "after-cinema food/drink transition",
  "answer": "yes",
  "confidence": 0.74,
  "evidence": [
    "User previously moved from cinema to restaurant twice",
    "Global transitions from cinema at night often go to restaurant/bar"
  ],
  "source_tools": ["CheckTransitionSupport"],
  "missing_evidence": [],
  "conflict": null
}
```

answer 只能是：

```text
yes
no
uncertain
not_available
```

其中：

- `uncertain` 表示证据有但不够强；
- `not_available` 表示数据集中没有该证据，不能推断。

### 9.4 不能 hallucinate 的规则

如果数据集没有营业时间，不能输出：

```text
“该店营业到 22:00”
```

只能输出：

```text
“历史数据中该 POI/category 曾在 21:00 附近被访问”
```

如果数据集没有评论，不能输出：

```text
“评论显示适合聚会”
```

只能输出：

```text
“评论证据不可用，当前判断基于类别、时间、距离和历史行为”
```

---

## 10. Need More Evidence 机制

### 10.1 什么情况下需要更多 evidence？

触发条件分为六类。

#### A. Intention uncertainty 高

```text
intention confidence < 0.6
```

处理：

```text
重新调用 peer behavior
扩大 query trajectory window
生成多个 sub-intentions
候选池按多个 intention 分支召回
```

#### B. Candidate coverage 不足

```text
filtered pool 没有覆盖 top intention categories
```

处理：

```text
调用 CategoryIntentRecall
扩大 spatial radius
调用 TemporalPopularityRecall
```

#### C. Top candidates 分数接近

```text
score(top1) - score(top2) < δ
默认 δ = 0.05
```

处理：

```text
对 top 5 调用 CompareTopCandidates
检查区分性 affordance：
  distance
  transition support
  same-hour revisit
  peer support
```

#### D. 关键 affordance 缺失

例如 intention 是 night food-and-drink，但候选没有 temporal evidence。

处理：

```text
调用 CheckTemporalFit
调用 TemporalPopularityRecall
调用 same-hour category profile
```

v1 如果有营业时间，则调用 CheckOpeningHours。

#### E. 证据冲突

例如：

```text
历史频率强烈支持 Starbucks
但当前 trajectory / peer behavior 支持 nightlife
```

处理：

```text
显式记录 conflict
降低冲突候选置信度
调用 CompareTopCandidates
必要时保留多样化 top results
```

#### F. 过度频率偏差

如果 top-10 全是历史最高频 POI，但 intention 指向不同类别：

处理：

```text
提高 category_intent_score 权重
引入 exploration candidates
调用 CategoryIntentRecall
```

这是我们比普通 one-shot LLM 推荐更想解决的问题。

---

## 11. Agent 需要哪些 evidence？

最终 agent 对每个推荐至少需要四类证据。

### 11.1 用户为什么有这个 intention？

证据来自：

```text
长期类别偏好
目标时间访问 pattern
最近轨迹
历史 transition
peer behavior，可选
```

示例：

```text
用户最近从 cinema 离开，目标时间是 21:00，历史上用户晚上常访问 restaurant 和 wine bar，因此推断当前 intention 是 evening food-and-drink。
```

### 11.2 候选 POI 为什么满足 intention？

证据来自：

```text
类别匹配
距离可行
目标时间 pattern 匹配
历史重访或相似类别重访
transition 支持
peer 支持
```

### 11.3 为什么它比其他候选更好？

证据来自 listwise comparison：

```text
候选 A 和 B 都是 restaurant，但 A 更近，且用户历史同小时访问过；
候选 C 是高频 POI，但类别与当前 intention 不匹配；
候选 D 类别匹配但距离超出用户常见移动范围。
```

### 11.4 哪些证据缺失？

必须显式输出：

```text
opening hours unavailable
reviews unavailable
photos unavailable
social situation unknown
price unavailable
```

这样用户和后续研究都知道当前 demo 的边界。

---

## 12. Final Ranking 输出需求

### 12.1 输出格式

Top-10 推荐列表：

```json
{
  "query_id": "q_0001",
  "user_id": "u_123",
  "target_time": "2012-07-04 21:00",
  "inferred_intention": {},
  "ranked_pois": [
    {
      "rank": 1,
      "poi_id": "p_88",
      "poi_name": "Barcibo Enoteca",
      "category": "wine bar",
      "alignment_score": 0.86,
      "confidence": 0.79,
      "reason": "Matches the inferred evening food-and-drink intention, is close to the last known cinema location, and fits the user's night-time wine bar pattern.",
      "supporting_evidence": [
        "category matches wine bar",
        "distance is 0.6 km from last known POI",
        "user often visits food/drink venues around 21:00",
        "similar category appeared in user's historical evening visits"
      ],
      "missing_evidence": [
        "opening hours unavailable",
        "review sentiment unavailable"
      ],
      "conflicts": []
    }
  ],
  "agent_trace_summary": []
}
```

### 12.2 UI 展示建议

demo 页面建议分五块：

```text
1. Query Context Card
2. Inferred Intention Card
3. Tool Trace Timeline
4. Candidate Pool Summary
5. Ranked POI Cards
```

每个 POI card 展示：

```text
Rank
POI name
Category
Distance
Alignment score
Confidence
Why recommended
Evidence chips
Missing evidence
Why not higher/lower，可选
```

---

## 13. Agent Loop 设计

### 13.1 v0 推荐采用有限自主 agent

不要一上来做完全自由 ReAct。建议：

```text
固定主流程
+
每个阶段允许 agent 在有限工具集中选择动作
+
最大 tool call 预算
```

原因：

- 可控；
- 方便 debug；
- 方便记录 trace；
- 避免 LLM 无限调用工具；
- 容易交给 Codex 实现。

### 13.2 Tool call 预算

v0 默认：

```text
max_total_tool_calls = 12
max_retrieval_tool_calls = 5
max_affordance_tool_calls = 8
max_reflection_rounds = 1
max_candidates_for_llm_ranking = 30
max_candidates_for_deep_comparison = 10
```

### 13.3 停止条件

满足任一条件即可停止：

```text
top1 confidence >= 0.75 且 top1-top2 margin >= 0.08
已经完成 1 轮 reflection
达到最大 tool call 预算
没有可用新 evidence
candidate pool 已覆盖 intention top categories
```

---

## 14. Demo 场景设计

### 14.1 场景 A：看完电影后的晚间餐饮

输入：

```text
历史：用户晚上常去 restaurant / bar
query trajectory：restaurant → bar → cinema
target time：21:00
last location：cinema
```

期望：

```text
intention = evening food-and-drink / social venue
候选 = 历史 food venues + cinema 附近 food/drink POI + transition recall
推荐 = 附近 wine bar / restaurant
```

展示重点：

```text
短期轨迹影响 intention
category + spatial + temporal affordance 对齐
```

### 14.2 场景 B：高频 POI 偏差纠正

输入：

```text
用户历史最高频是 coffee shop
但 query trajectory 和 target time 指向 dinner/nightlife
```

期望：

```text
agent 不直接推荐最高频 coffee shop
而是识别当前 intention，召回 restaurant/bar
```

展示重点：

```text
反 one-step frequency bias
```

### 14.3 场景 C：稀疏用户

输入：

```text
用户历史很短
query trajectory 只有 1-2 个 check-ins
```

期望：

```text
agent 调用 peer behavior / temporal popularity
intention confidence 降低
推荐理由明确说用户历史不足
```

展示重点：

```text
缺证据时如何补证据
```

### 14.4 场景 D：候选池不足触发扩召回

输入：

```text
initial candidate pool 缺少 intention top category
```

期望：

```text
DetectEvidenceGap → ExpandCandidatePool → CategoryIntentRecall
```

展示重点：

```text
agentic behavior，而不是固定 pipeline
```

---

## 15. 模块级需求拆分

### Module 1：Context Builder

输入：

```text
raw check-ins
POI catalog
user_id
query trajectory
target time
```

输出：

```text
Context Snapshot
User Mobility Stats
Short-term Trajectory Summary
Dataset Capabilities
```

验收：

```text
能输出用户 top categories、top POIs、hour pattern、day pattern、distance pattern、recent trajectory summary。
```

### Module 2：Intention Engine

输入：

```text
Context Snapshot
User Mobility Stats
Peer Summary，可选
```

输出：

```text
structured intention JSON
```

验收：

```text
每个 intention 至少包含 likely_categories、spatial preference、temporal preference、evidence、confidence。
```

### Module 3：Tool Planner

输入：

```text
Intention JSON
Context Snapshot
Dataset Capabilities
```

输出：

```text
tool plan JSON
```

验收：

```text
能根据不同 intention 选择不同 recall tools；
能说明每个工具调用原因；
不能调用数据集不支持的工具。
```

### Module 4：Candidate Manager

输入：

```text
tool plan
POI catalog
user history
```

输出：

```text
raw candidate pool
filtered candidate pool B=30
candidate source labels
```

验收：

```text
每个候选记录来源：
historical / spatial / category / transition / peer / popularity
```

### Module 5：Affordance Builder

输入：

```text
filtered candidates
intention
context
```

输出：

```text
affordance profile for each candidate
```

验收：

```text
每个候选至少有 category、spatial、temporal、revisit、transition 五类 affordance。
```

### Module 6：Alignment Ranker

输入：

```text
intention
affordance profiles
```

输出：

```text
ranked candidates
alignment scores
confidence
```

验收：

```text
输出 Top-10；
每个推荐必须有 score decomposition。
```

### Module 7：Reflection Controller

输入：

```text
ranked candidates
evidence gaps
conflicts
```

输出：

```text
是否继续调用工具；
若继续，调用哪些工具；
若停止，停止原因。
```

验收：

```text
能处理 top score 接近、候选覆盖不足、关键证据缺失、证据冲突四种情况。
```

### Module 8：Explanation Renderer

输入：

```text
final ranking
intention
affordance profiles
agent trace
```

输出：

```text
human-readable explanation
structured JSON
```

验收：

```text
每个推荐都能解释：
用户想做什么；
该 POI 满足什么；
缺少什么证据；
为什么排在这个位置。
```

---

## 16. 关键产品原则

### 16.1 Evidence first

Agent 不能凭空说 POI 适合某种场景。所有判断必须来自工具 evidence。

### 16.2 Dataset-aware reasoning

如果数据没有评论、图片、营业时间，就不能生成相关断言。

### 16.3 Intention drives tool use

Intention 不是装饰性解释，而是决定召回、证据获取和排序权重的控制状态。

### 16.4 Affordance is candidate-specific

不能只说“用户想去餐厅”，还要说明“这个 POI 为什么能满足这个意图”。

### 16.5 Reflection is conditional

不是每次都反思。只有在证据不足、分数接近、冲突或候选覆盖不足时触发。

---

## 17. v0 到 v2 迭代路线

### v0：结构化轨迹版

数据：

```text
Foursquare NYC/TKY
POI category + timestamp + coordinates + user trajectories
```

能力：

```text
intention inference
candidate tools
structured affordance
alignment ranking
reflection once
explanation
```

不做：

```text
reviews
images
opening hours
price
real-time signals
```

### v1：评论增强版

数据：

```text
Yelp Open 或小规模 Yelp-style subset
```

新增工具：

```text
SearchReviews
SummarizeRecentReviews
ExtractReviewAffordance
DetectReviewTemporalConflict
```

新增 affordance：

```text
quietness
group_friendly
service_quality
reservation_need
social_atmosphere
work_friendly
```

v1 不一定用于主流 next POI benchmark，可以作为 richer demo。

### v2：图像/外部 API 版

新增：

```text
photo captions
visual layout
crowd density
seating structure
opening hours
price
```

这时才更接近 Agent4POI 的 multimodal affordance reasoning。

---

## 18. 风险与规避

### 风险 1：候选召回决定上限

如果 ground-truth 不在候选池，后面怎么推理都没用。

规避：

```text
多路召回
候选覆盖检查
保留 exploration candidates
记录 candidate coverage 指标
```

### 风险 2：LLM hallucination

规避：

```text
结构化工具输出
dataset capability guardrail
每个结论必须绑定 evidence
不允许凭空补营业时间/评论
```

### 风险 3：过度依赖历史重访

规避：

```text
CategoryIntentRecall
SpatialRecall
TransitionRecall
ExplorationFit
PeerRecall
```

### 风险 4：agent 太自由导致不稳定

规避：

```text
有限工具集
固定状态机
最大 tool call budget
JSON action schema
失败 fallback
```

### 风险 5：v0 affordance 看起来不够“丰富”

规避：

```text
明确 v0 是 structured mobility affordance
把评论/图像作为 v1/v2 插件
先证明 workflow，而不是一次性做全多模态
```

---

## 19. v0 验收标准

### 功能验收

1. 输入一个 query trajectory，系统能输出 Top-10 POI。
2. 输出必须包含 intention card。
3. 输出必须包含候选池来源统计。
4. 输出必须包含每个 Top-10 POI 的 affordance profile。
5. 输出必须包含 agent trace。
6. 至少支持一次 reflection。
7. 不允许输出数据集中不存在的证据类型。

### 质量验收

1. 每个推荐理由至少包含 3 条 evidence。
2. 每个候选的 score decomposition 可查看。
3. 当 evidence 缺失时，系统明确标记 missing evidence。
4. 当 top candidates 分数接近时，触发 comparison 或 reflection。
5. 当用户历史稀疏时，intention confidence 会下降，并触发 peer/global evidence。

### Demo 验收

至少准备 4 个 replay cases：

```text
Case A: 正常晚间餐饮预测
Case B: 纠正历史高频偏差
Case C: 稀疏用户
Case D: 候选不足触发扩召回
```

---

## 20. 最终建议的 demo 形态

v0 demo 不要做成“黑盒 API 返回 Top-10”，而是做成一个可观察的 agent workspace：

```text
左侧：用户轨迹时间线
中间：agent steps / tool trace
右侧：intention、candidate pool、affordance profiles、final ranking
```

这样最能展示我们的核心价值：

> 不是 LLM 猜中了哪个 POI，而是它如何基于 intention 主动找候选、查证据、构建 affordance、对齐排序，并在证据不足时补充推理。

这版 demo 的核心创新可以命名为：

> **IAA-AgentPOI: Intention-Affordance Aligned Agent for Next POI Recommendation**

更短一点：

> **IAR-POI: Intention-Affordance Reasoning for Agentic POI Recommendation**

---

## 21. 下一步建议交付物

下一步建议先锁定三份交付物：

```text
1. Context Snapshot JSON schema
2. Tool Registry & Tool Output schema
3. 一个完整 replay case 的端到端样例输出
```

这三份交付物完成后，就可以进入 Codex 实现阶段。
