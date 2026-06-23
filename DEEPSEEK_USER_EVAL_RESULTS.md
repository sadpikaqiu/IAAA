# DeepSeek 用户级 Session Eval 结果

## 测试设置

- 数据：`datasets/NYC`
- 切分：每个用户按时间排序，前 80% 作为长期历史，后 20% 中原始 `trajectory_id` 的最后一次签到作为 session target。
- LLM：DeepSeek
- Trace 保存：`--save-runs`
- 修复点：候选召回允许使用全量 POI 元数据作为候选宇宙，但 POI 访问次数和流行度仍只来自可见历史，避免把未来 visit count 泄漏进排序。

## 用户 1

命令：

```powershell
python -m iaa_agent evaluate --user-id 1 --llm deepseek --save-runs outputs/eval_runs/user_1_deepseek --out outputs/evaluation/user_1_deepseek.json
```

结果：

```text
total = 2
Hit@1 = 0.0
Hit@5 = 0.0
Hit@10 = 0.0
MRR = 0.0
```

解释：用户 1 的 held-out session 很少，且目标类别和历史强信号不一致。这个用户更适合当失败案例分析，不适合判断系统整体是否有效。

## 用户 1005

命令：

```powershell
python -m iaa_agent evaluate --user-id 1005 --llm deepseek --save-runs outputs/eval_runs/user_1005_deepseek --out outputs/evaluation/user_1005_deepseek.json
```

结果：

```text
total = 1
Hit@1 = 0.0
Hit@5 = 0.0
Hit@10 = 0.0
MRR = 0.0
```

解释：只有 1 条测试 session，样本太少，指标不稳定。

## 用户 349

命令：

```powershell
python -m iaa_agent evaluate --user-id 349 --llm deepseek --save-runs outputs/eval_runs/user_349_deepseek_v2 --out outputs/evaluation/user_349_deepseek_v2.json
```

结果：

```text
total = 10
Hit@1 = 0.0
Hit@5 = 0.4
Hit@10 = 0.7
NDCG@5 = 0.194846
NDCG@10 = 0.297133
MRR = 0.173571
```

DeepSeek usage：

```text
prompt_tokens = 40509
prompt_cache_hit_tokens = 19712
prompt_cache_miss_tokens = 20797
completion_tokens = 6037
total_tokens = 46546
```

解释：用户 349 有较丰富历史，session 目标也更符合长期 routine / spatial / transition 证据，因此 Hit@10 达到 0.7，超过当前目标 0.5。

## 当前结论

- 用户 349 证明系统在有足够历史证据时可以工作。
- 用户 1 和 1005 暴露了稀疏用户/少样本用户的困难。
- 后续如果要提升全局指标，需要重点增强 sparse-user recall，而不是只调 DeepSeek intention。

