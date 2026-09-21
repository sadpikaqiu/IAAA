# 固定 Pipeline 多模态消融（2026-09-17）

目的：在前一轮定位到召回加分、候选池扩展和意图类别变化等机制后，量化它们各自的影响。这里是验证集组件诊断，不替代原全量测试，也不会根据测试结果自动选最优配置。

## 数据与对照协议

- 每位用户前 70% 事件作为个人和全局统计历史，验证目标为该用户 `[70%,80%)` 区间内的会话末次签到；原 `[80%,100%)` 测试会话全部排除。可见历史和会话上下文须严格早于目标时间。
- NYC、TKY 各按会话 ID 的固定哈希顺序抽 500 个会话。抽样不使用目标 POI、原模型命中情况或证据覆盖情况。样本固定后不替换失败会话。
- 沿用原先的逐用户时间划分，而非统一日历时间切分；静态 POI 图片和评论仍存在观察日期未知的限制。
- 每个会话分别请求文本、图文、仅图片、仅评论四个意图。每城固定 100 个会话额外重复文本与图文请求，用来估计相同提示的模型波动。正常共 4,400 次本地模型请求。
- 使用同一 Qwen 服务、非 thinking、temperature=0、seed=42、max_tokens=4096；上下文尾部 K=5。配置对比复用已接受的意图，所有下游重放禁止调用模型。
- 仅图片和仅评论使用同一联合 TF-IDF 索引的模态遮罩，不重新拟合词表/IDF；保留每个模态原来的 0.05 排序系数，不把留下的模态权重翻倍。
- 每个意图最多尝试三次；按统一规则接受 schema 合法、usage 存在且正常停止的返回。保存原始返回和字段错误，禁止把 heuristic fallback 混入结果。全部失败时明确报告，不删除或换掉会话。

## 预先固定的配置

`text_budget` 表示复用同会话文本基线的扩展决定，固定 30/60 候选预算。该控制仅用于归因分析；`mobility_scores` 则只在扩展判据中去掉图片/评论分数，无需额外请求文本模型。

| 配置 | 意图 | 下游证据变化 | 扩展决定 |
| --- | --- | --- | --- |
| text | 文本 | 无 | 原自动规则 |
| intent_only_both | 图文 | 无 | text_budget |
| intent_only_images | 图片 | 无 | text_budget |
| intent_only_reviews | 评论 | 无 | text_budget |
| recall_only | 文本 | 保留证据召回、prior 加分和保底名额，去掉排序加分 | text_budget |
| rank_only | 文本 | 仅排序加分；候选池必须与文本基线完全一致 | text_budget |
| full_both | 图文 | 原完整图文流程 | 原自动规则 |
| no_quota | 图文 | 仅去掉证据保底名额 | 原自动规则 |
| no_prior_boost | 图文 | 仅去掉归一化的证据 prior 加分 | 原自动规则 |
| no_rank_bonus | 图文 | 仅去掉证据排序加分 | 原自动规则 |
| mobility_reflection | 图文 | 原完整图文流程 | mobility_scores |
| fixed_reflection | 图文 | 原完整图文流程 | text_budget |
| no_prior_fixed_reflection | 图文 | 去掉 prior 加分，保留名额和排序 | text_budget |
| no_prior_no_quota_fixed_reflection | 图文 | 再去掉名额 | text_budget |
| images_only | 图片 | 仅图片证据 | text_budget |
| reviews_only | 评论 | 仅评论证据 | text_budget |
| text_repeat / full_both_repeat | 相同提示的独立重复返回 | 各自原流程 | 原自动规则 |

前 16 组每城 500 个会话；最后两组每城 100 个。报告 Hit@1/5/10、NDCG@10、MRR@10、原始/最终候选召回、IH/OOH、扩展比例、配对获益/损失以及按用户聚类的 bootstrap 区间。区间是探索性的，未进行多重比较校正。跨版本对照不能用只有部分会话完成的中途统计下结论。

## 执行与复核

入口为 `scripts/run_multimodal_ablations.py`，实验控制位于 `scripts/mm_ablation_support.py`。生产模块保持原冻结版本不变。

```bash
python scripts/run_multimodal_ablations.py \
  --source-experiment /home/yzj/IAAA/outputs/experiments/mm_pipeline_20260914 \
  --data-root /home/yzj/IAAA/datasets \
  --output-dir /home/yzj/IAAA/outputs/experiments/mm_ablation_20260917/results
```

执行顺序是 NYC 4 会话预检 → NYC 500 → TKY 4 会话预检 → TKY 500；预检会话属于各城固定 500 个样本，不会额外计入指标。预检同时验证实际缓存意图下实验包装器与原生产引擎等价。验证通过后自动继续。

进度在 `progress.json`，协议和 SHA256 在 `manifest.json`，意图缓存及每次尝试在 `intentions/`，逐会话配置结果在 `cases/`，城市汇总在 `summaries/`。同命令支持断点续跑；已完成调用和配置不会重跑，进程锁防止重复运行。若配置、代码、数据或关键依赖版本改变，拒绝复用原结果。

无需轮询等待整个实验结束。使用服务器 `screen` 持续运行，完成后读取各城市汇总和质量门槛，再决定是否需要调整方案或开展新的独立测试。
