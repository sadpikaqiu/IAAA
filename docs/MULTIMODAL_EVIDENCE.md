# 在固定推荐流程中使用图片和评论

NYC/TKY 的原有 CSV 继续负责轨迹、历史划分、POI 类别和坐标。WWW2024 的
`review_summary.json`（实际为 JSONL）及 Flash 生成的图片证据按原始 Foursquare
POI ID 连接。未指定 `--evidence-snapshot` 时保留原有文本流程。

## 接入位置

1. **意图推断**：在原有的一次文本模型请求里附上最近 5 个可见 POI 的证据。
   每个 POI 每种模态最多 1 段、每段 350 字符。只读取已访问地点，隐藏目标
   不参与这一步。外部文本被标记为资料，不能作为指令执行。
2. **补充召回**：使用推断出的活动和类别检索 POI 内容。候选仍限定在原有
   POI 目录；默认距最后可见位置 10 km，最多补充 30 个候选，最终池中为
   内容召回预留最多 5 个名额。固定反思轮次可将半径和召回数量扩大一倍。
3. **排序与解释**：增加 `review_intent_relevance` / `image_intent_relevance`。
   权重总量默认 0.10；缺失模态贡献 0，不产生否定设施判断。每个引用保留
   评论行号/索引或图片编号、文件路径和 SHA-256，可从完整 trace 回查。

这是固定控制流程，没有增加自主工具选择，也没有增加在线图片模型调用。
检索和打分暂用字符 n-gram TF-IDF，是**词面相关性**，不是语义嵌入、用户偏好
概率或设施真实性判断。它可以作为接入基线，但英文摘要与日文评论之间没有
可靠的跨语言匹配能力，TKY 效果需要单独验证。

仅使用图片产物中的 `visual_evidence`；不把 `possible_activities` 或联合摘要中
未绑定图片的推断作为事实。评论保留为访客报告，视觉证据保留为模型观察。
两者的观察日期未知，实验必须标记为使用静态外部 POI 知识，不能声称资料在
2012–2013 年预测时已经可用。无图片/无评论的 POI 仍可通过原有各路召回进入候选。

## 图片处理完后冻结输入

在项目根目录运行：

```powershell
# 快速查看缺失/无效数量；不会写快照，也不会调用 API
python scripts/prepare_poi_evidence.py --city both --check-only

# 完成后校验原图字节、生成配置、引用编号并生成不可覆盖的快照
python scripts/prepare_poi_evidence.py --city both
```

快照输出为 `outputs/poi_evidence/NYC_<hash>.json` 和 `TKY_<hash>.json`。
使用打印出来的实际路径。准备程序不会改动 CSV、原图或图片摘要。

有原图的目录 POI 必须具有有效摘要或经过审计的不可用记录，生成配置需要一致，
且不能存在正在写入的 `.writer.lock`。尚未处理的图片、错误/过期摘要或混合配置会阻止冻结。完整性
只针对 CSV 中的 POI；原本没有图片的 POI 不要求生成摘要。全部图片都被判为
不适合提供场所证据时，允许成功产物中的视觉证据为空，并如实记录覆盖率。

生成器保留损坏文件的路径、哈希和解码错误，使用同组可读图片生成摘要；快照会
重新校验这些排除项。服务端明确拒绝的图片可通过 `--unavailable-manifest <JSON>`
传入审计名单，要求错误文件哈希、原图指纹和 `data_inspection_failed` 记录匹配。
这种 POI 保留评论和候选资格，仅缺少视觉证据；网络或格式错误不能借此跳过。
`complete` 表示全部原图 POI 已有明确处理结果，并不表示每个 POI 都有可用图片。

`--check-only` 不逐张读取原图字节，不是正式的完整性认证。`--allow-partial`
仅供开发导出；推荐入口会拒绝这种快照。评估读取固定快照，验证内容哈希、城市
与 CSV 哈希，不会随着后台图片生成改变输入。服务器运行时需复制同一份快照和
相同 CSV，并部署对应代码；在线推荐无需复制原图或 Flash API Key。

## 同会话对照

以下命令仅为图片完成后的执行步骤。把 `$snapshot` 设为准备程序输出的 NYC
快照，模型地址使用实际服务器配置；`--smoke-limit 50` 仅为试跑。

```powershell
$snapshot = 'outputs/poi_evidence/NYC_<hash>.json'
$common = @('evaluate', '--data-dir', 'datasets/NYC', '--variant', 'p4v1',
  '--llm', 'openai', '--model', 'Qwen/Qwen3.8-27B-FP8',
  '--base-url', 'http://10.18.32.109:8000/v1', '--no-thinking',
  '--llm-max-tokens', '4096', '--concurrency', '4',
  '--intention-context-size', '5', '--smoke-limit', '50',
  '--report-stratified', '--report-candidates', '--no-allow-fallback')
python -m iaa_agent @common --out outputs/evaluation/text_smoke.json
python -m iaa_agent @common --evidence-snapshot $snapshot --evidence-mode both --out outputs/evaluation/image_review_smoke.json
```

两组都显式使用 `--intention-context-size 5`，避免短期上下文长度差异混入图文
效果。未指定该参数时，纯文本保持旧行为，图文默认截取最近 5 条。长期统计不变。
可用 `--evidence-mode reviews` / `images` 做模态消融，其他配置保持相同。
正式全量对照去掉两条命令共同参数里的 `--smoke-limit 50`。TKY 使用其对应 CSV
目录和快照单独运行。

检查输出中：

- `run_config`、模型参数、`evidence_snapshot.snapshot_id` 和覆盖数量。
- `candidate_diagnostics.sessions` 中完全相同的 `(user_id, trajectory_id)`，
  以及逐会话 rank、预测列表；标签只在预测结束后用于统计。
- `CandidateRecall`、筛选前召回的 `RawCandidateRecall`，尤其是 OOH 子集。
- IH/OOH 的 Hit@1/5/10、NDCG 和 MRR。
- `fallback_count=0`、`usage_missing_count=0`、`all_sessions_used_llm=true`。
  `--no-allow-fallback` 只标记违反条件的会话，不会自动将混合结果变成正式指标。

没有通过用量/回退检查的运行或使用 `--llm fake` 的功能试跑不能作为正式模型
效果证据。当前开发验证使用小型合成数据测试 ID 连接、引用、缺失处理、目标
隔离和串行/并发一致性；真实图文效果需等图片完成后再测。

## 服务器连续测试

`scripts/evaluate_fixed_pipeline.py --full` 按 NYC smoke → NYC full → TKY smoke →
TKY full 执行，每阶段均含文本和图文两组。每个城市固定同一组原始会话；smoke
用会话 ID 的确定性哈希选择 50 个会话，全量使用该城市全部合格会话。结果保存
会话清单、逐会话预测、候选召回、用量和配置。任一组出现 fallback、缺少用量或
会话不匹配，就保存已有结果并停止后续阶段，不将错误混入正式比较。

2026-09-14 本次测试在 109 的独立目录
`/home/yzj/IAAA/outputs/experiments/mm_pipeline_20260914/` 中使用冻结代码和快照，
复用哈希已核对的服务器 CSV，不覆盖原仓库或旧实验结果。`results/progress.json`
记录当前阶段、完成会话数和失败原因，各阶段的 `paired.json` 保存对照结果。
