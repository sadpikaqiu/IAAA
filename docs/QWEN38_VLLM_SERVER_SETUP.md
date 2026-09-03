# Qwen3.8-27B-FP8 + vLLM 部署说明

## 1. 已验证环境

IAA-Agent 通过 vLLM 的 OpenAI-compatible API 调用本地
`Qwen3.8-27B-FP8`。当前服务器已验证配置如下：

- Conda 环境：`iaaa`，Python 3.11。
- GPU：NVIDIA RTX 6000D，约 83 GiB 显存。
- PyTorch：`2.13.0+cu130`。
- vLLM：`0.28.0`。
- Transformers：`5.16.1`。
- 本地模型目录：`~/Model/Qwen38-27B`。
- OpenAI API 模型名：`Qwen/Qwen3.8-27B-FP8`。

模型 checkpoint 为 block-wise FP8，实测文件大小约 28.75 GiB，权重加载后
占用约 27.64 GiB 显存。相关上游资料：

- https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

## 2. 安装

激活项目环境后，在仓库根目录执行：

```bash
conda activate iaaa
cd ~/IAAA
bash scripts/setup_qwen38_vllm.sh
```

安装脚本默认使用清华 PyPI 镜像，并将 vLLM、Transformers 和 IAA-Agent
安装到当前环境。镜像或版本可以显式覆盖：

```bash
PIP_INDEX_URL=https://pypi.org/simple VLLM_VERSION=0.28.0 \
  bash scripts/setup_qwen38_vllm.sh
```

脚本不会下载模型；启动前应保证 `~/Model/Qwen38-27B/config.json` 存在。

## 3. 使用 screen 启动

默认启动参数针对共享 GPU：显存预算 0.60、16K context、最多 16 个序列，
同时使用 BF16 KV cache。checkpoint 没有提供已校准的 FP8 KV scale，因此正式
实验不默认启用 FP8 KV cache，以避免潜在精度损失。

```bash
conda activate iaaa
cd ~/IAAA
screen -L -Logfile ~/qwen38-vllm.log -dmS iaaa-qwen38 \
  bash scripts/serve_qwen38_vllm.sh
```

查看会话和日志：

```bash
screen -ls
tail -f ~/qwen38-vllm.log
```

独占 GPU 时可以提高显存预算和并发：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.90 VLLM_MAX_NUM_SEQS=32 \
  bash scripts/serve_qwen38_vllm.sh
```

脚本默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`。当前服务器没有 CUDA Toolkit
中的 `nvcc`，关闭该采样器可避免 FlashInfer 首次 JIT 编译失败；模型 FP8
矩阵乘仍使用 CUTLASS 内核。服务同时启用 `--reasoning-parser qwen3`，非思考
请求不受影响，思考请求的 `reasoning_content` 与最终 JSON 会被分开返回。

## 4. 服务检查

```bash
cd ~/IAAA
python scripts/smoke_vllm.py
```

成功时输出包含模型名、合法 JSON 以及 usage。也可以检查健康端点：

```bash
curl -i http://127.0.0.1:8000/health
```

## 5. 接入 IAA-Agent

先跑一个 session 并保存完整 trace：

```bash
cd ~/IAAA
python -m iaa_agent evaluate \
  --data-dir datasets/NYC \
  --user-id 349 \
  --smoke-limit 1 \
  --save-runs outputs/runs/qwen_smoke \
  --out outputs/evaluation/qwen_smoke.json \
  --llm openai \
  --variant p4v1 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --no-allow-fallback
```

`openai` 模式默认使用占位 API key `EMPTY`，本地无鉴权 vLLM 服务不需要真实
密钥。正式全量评估前，建议先跑 20 至 50 个 session：

```bash
python -m iaa_agent evaluate \
  --data-dir datasets/NYC \
  --smoke-limit 50 \
  --llm openai \
  --variant p4v1 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 8 \
  --report-stratified \
  --out outputs/evaluation/qwen38_smoke50.json
```

确认 `fallback_count=0`、`usage_missing_count=0` 后，再去掉 `--smoke-limit`
进行全量评估。客户端 concurrency 控制并行 HTTP 请求数，实际吞吐还取决于
vLLM batching、prompt 长度以及 GPU 上是否存在其他任务。

## 6. 思考模式对照实验

思考模式必须显式开启，并为推理过程预留更大的 completion budget：

```bash
python -m iaa_agent evaluate \
  --data-dir datasets/NYC \
  --smoke-limit 50 \
  --llm openai \
  --variant p4v1 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --thinking \
  --reasoning-effort medium \
  --llm-max-tokens 4096 \
  --concurrency 8 \
  --no-allow-fallback \
  --report-stratified \
  --out outputs/evaluation/qwen38_thinking_smoke50.json
```

思考请求不能同时使用从首 token 生效的 JSON grammar，否则模型无法完成 think
block；客户端会改用 prompt 约束并解析最终 JSON。先比较 50-session 的 JSON
成功率、reasoning/completion tokens、吞吐和推荐指标，再决定是否运行全量思考
版。`xhigh` 可能显著增加延迟，初始对照推荐 `medium + 4096 tokens`。

## 7. 验收标准

1. `setup_qwen38_vllm.sh` 完成后 `pip check` 和 CUDA import 均通过。
2. `/health` 返回 HTTP 200，`/v1/models` 包含预期模型名。
3. `smoke_vllm.py` 返回合法 JSON 和非空 usage。
4. IAA-Agent smoke test 的 `llm_mode` 为 `openai`，且没有 fallback。
5. 每个需要审计的 session 都生成完整 `AgentRunResult` trace。
