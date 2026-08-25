# DeepSeek 输入缓存说明

## 调研结论

DeepSeek 官方 API 的 context caching 是前缀缓存机制。它不需要像 Anthropic Claude 那样在请求里显式标注 `cache_control`；只要多次请求的输入前缀完全一致，DeepSeek 会以 best-effort 方式复用缓存。

官方文档给出的关键判断字段在响应的 `usage` 中：

```text
prompt_cache_hit_tokens
prompt_cache_miss_tokens
```

其中：

- `prompt_cache_hit_tokens` 表示本次输入中命中缓存的 token 数。
- `prompt_cache_miss_tokens` 表示本次输入中没有命中缓存的 token 数。

OpenCode 相关讨论里，主要问题不是 DeepSeek 不支持缓存，而是一些 agent 工具没有正确展示或统计 DeepSeek 返回的 cache usage 字段，导致用户看到 cache read/write 一直为 0。

参考：

- DeepSeek Context Caching: https://api-docs.deepseek.com/guides/kv_cache
- DeepSeek Chat Completion usage fields: https://api-docs.deepseek.com/api/create-chat-completion
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing
- OpenCode cache usage issue: https://github.com/anomalyco/opencode/issues/24189

## 我们的实现

当前实现做了三件事：

1. `DeepSeekClient` 解析每次响应中的 usage 字段。
2. `AgentRunResult.agent_trace_summary` 中记录每次 DeepSeek intention call 的 cache hit/miss tokens。
3. `evaluate --save-runs` 的 summary 中汇总 `llm_usage`，并给每条 session 记录 `deepseek_usage`。

此外，prompt 结构被调整为更适合前缀缓存：

```text
system message: 固定角色说明
user message 1: 固定 IAA-Agent intention schema 和 guardrails
user message 2: 同一用户稳定长期画像
user message 3: 当前 session 的动态上下文
```

这样，在 `evaluate --user-id 349` 这种单用户多 session 评估里，前两个消息和长期画像可以被多次复用，后续 session 的请求会更容易命中缓存。

## 如何查看缓存效果

示例命令：

```powershell
python -m iaa_agent evaluate --user-id 349 --llm deepseek --save-runs outputs/eval_runs/user_349_deepseek
```

查看 summary：

```json
{
  "llm_usage": {
    "prompt_tokens": 40509,
    "prompt_cache_hit_tokens": 19712,
    "prompt_cache_miss_tokens": 20797,
    "completion_tokens": 6037,
    "total_tokens": 46546
  }
}
```

查看单条 trace 的 intention 阶段：

```text
DeepSeek cache hit/miss tokens: 1152/1277
```

## 注意事项

- DeepSeek 缓存是 best-effort，不保证每次 100% 命中。
- 缓存构建需要时间，刚开始的请求可能 miss 较多。
- 缓存通常会在数小时到数天内自动清理。
- 对全量多用户评估来说，固定 schema 前缀会重复命中；对单用户评估来说，稳定长期画像也能参与缓存。
- 当前缓存不是“复用上一次回答”，而是“复用输入前缀的 KV cache”；模型仍然会生成新的输出。

