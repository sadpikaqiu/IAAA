# WWW2024 POI 图片联合摘要

脚本 `scripts/summarize_poi_images.py` 调用千问 AI 平台的 `qwen3.8-flash`，
把同一 POI 的多张图片转换为可供文本推荐模型使用的结构化视觉证据。
这是独立预处理步骤；推荐 agent 可通过显式证据快照加载已完成的结果，详见
[固定流程图文接入说明](MULTIMODAL_EVIDENCE.md)。

## 安装与运行

在项目根目录运行，Python 3.11+，只需额外安装 Pillow：

```powershell
python -m pip install -e ".[vision]"

# 仅统计两个城市的数据，不读取 Key、不调用 API、不生成文件
python scripts/summarize_poi_images.py --city both --dry-run

# 先验证 2 个 POI；默认英文摘要，适合现有英文 POI 元数据
python scripts/summarize_poi_images.py --city NYC --limit 2 --workers 1

# 全量处理两个城市；默认跳过已成功且输入/配置一致的 POI
python scripts/summarize_poi_images.py --city both --workers 4
```

中断后执行相同命令即可继续。`--limit N` 是按 POI ID 排序后的前 N 个 POI，
在跳过缓存之前应用；例如反复执行 `--limit 2` 会检查同样的两个地点。
原始图片只读，转换在内存中完成；压缩图和 Base64 不会写入输出。

本机可用的 Conda Python 为 `C:\miniconda3\python.exe`。如果终端中的 `python`
指向其他环境，可以用该绝对路径替代上面命令的 `python`。

## API 配置

按顺序使用 `QWEN_API_KEY`、`DASHSCOPE_API_KEY`，否则读取项目根目录
`API keys.txt`。脚本只选择标签含 `qwen` / `dashscope` / `千问` / `百炼` 的条目，
支持中文冒号和带点号的新格式 Key。当前的 `qwen38-plus` 标签可直接识别，
标签不决定调用模型，默认模型始终是 `qwen3.8-flash`。
若文件里有多个不同 Qwen Key，使用 `--key-label qwen38-plus` 明确选择。
Key 不会写入摘要、日志或命令行；环境变量优先于文件选择。

千问 AI 平台通用 API 的默认地址为
`https://dashscope.aliyuncs.com/compatible-mode/v1`，采用 `POST /chat/completions`。
图片使用 `image_url` 的 Base64 data URL；直接发送 HTTP 时，
`enable_thinking: false` 放在请求顶层，同时请求 JSON 输出。
可通过 `--base-url` / `QWEN_BASE_URL`、`--model` / `QWEN_VISION_MODEL` 修改。
这里不读取推荐服务器的 `OPENAI_BASE_URL`，避免配置互相影响。

接口依据：[千问 API Key 文档](https://platform.qianwenai.com/docs/api-reference/preparation/api-key)、
[Qwen3.8-Flash 文档](https://platform.qianwenai.com/docs/developer-guides/getting-started/latest-model)、
[图像输入与多图理解](https://help.aliyun.com/zh/model-studio/vision)。

## 数据映射与覆盖

预期目录：

```text
datasets/NYC_WWW2024/NYC/image/downloaded_multimodal_data/gmap_<row>_<poi_id>_<index>.png
datasets/TKY_WWW2024/TKY/image/downloaded_multimodal_data/gmap_<row>_<poi_id>_<index>.png
```

脚本递归扫描 `image/`，取文件名中的 24 位原始 Foursquare POI ID 归组，
图片编号按数值排序。不能解析的图片会报错，避免悄悄漏掉输入。
2026-09-13 本地盘点：NYC 45,337 张图片，涉及 3,598 个 POI；
TKY 89,136 张图片，涉及 6,643 个 POI；每个 POI 最多 16 张。
这些 POI ID 均能匹配各自原始轨迹文件，但并非所有轨迹 POI 都有图片。

默认对同一 POI 的全部唯一图片做一次联合识别。以文件 SHA-256 去除完全重复
的图片，输出保留每个原始文件及重复关系；没有图片的 POI 不生成虚构摘要。
`.png` 后缀可能实际是 JPEG，Pillow 按真实内容解码，然后统一为 JPEG 上传。
默认长边最多 1024 像素、质量 85，不放大小图片。

可选控制：

```powershell
# 输出中文摘要（修改语言会使对应缓存失效）
python scripts/summarize_poi_images.py --city NYC --limit 2 --language zh

# 指定一个原始 POI ID
python scripts/summarize_poi_images.py --city NYC --poi-id 49bbd6c0f964a520f4531fe3

# 明确抽样以降低用量：最多 8 张，按图片顺序均匀抽取，并保留未选图片记录
python scripts/summarize_poi_images.py --city both --max-images-per-poi 8
```

默认 `--max-images-per-poi 0` 表示不抽样；不要把人为抽样结果描述为全部图片的总结。
单请求序列化后超过 9 MB 会明确报错，此时降低 `--max-edge` 或显式限制图片数。
当前最多 16 张的小图不需要分批汇总。

## 输出与断点续跑

```text
outputs/poi_image_summaries/
  NYC/pois/<poi_id>.json       # 成功的结构化摘要
  NYC/errors/<poi_id>.json     # 失败原因与已收到的用量
  TKY/pois/<poi_id>.json
  TKY/errors/<poi_id>.json
  runs/<run_id>.json           # 每次运行的统计、失败列表、用量
```

每个成功文件包含：

- `city`、`poi_id`：原始数据标识。
- `result.summary`：跨图片联合摘要。
- `result.visual_evidence`：直接可见的证据及 `image_indices`。
- `result.possible_activities`：由可见物体支持的可能活动，单独标记为推断。
- `result.uncertainties`、`result.image_notes`：局限及每张图片的内容/相关性。
- `images`：原始相对路径、SHA-256、重复关系、被送入模型的编号和实际格式。
- `config`、`input_fingerprint`：模型、端点、提示词版本、语言与图像处理配置。
- `attempts`：每次请求的状态、响应模型、完成原因、耗时和 API 返回的用量。

把 `result` 或 `result.summary` 按原始 `poi_id` 关联到后续 POI profile 即可。
`image_indices` 对应 `images[].image_index`，可以追溯原图。
原始文件路径相对于该城市的 `datasets/<CITY>_WWW2024/<CITY>/`。

只有输入文件内容和生成配置一致、摘要结构完整的成功记录才会命中缓存。
配置或图片变化会自动重新生成；`--force` 强制刷新。每个 POI 使用临时文件加
原子替换保存，Ctrl+C 时等待已开始的请求保存结果。强制刷新失败时保留上次
成功文件，并另写 `errors/`；使用产物时应核对其配置和对应运行报告。
不同实验配置若需要同时保留，请分别指定 `--output-dir`。

同一个输出目录一次只能启动一个写入进程。正常退出自动移除 `.writer.lock`；
若进程被强制终止，确认该目录已无脚本进程运行后再手动移除残留锁文件。

默认 2 个 worker、所有 worker 共享最小 0.5 秒请求启动间隔、120 秒超时。
429、临时网络错误和 5xx 使用指数退避，默认最多重试 3 次；JSON/字段校验失败时
带上具体校验反馈请求模型重新生成，保留失败尝试的用量，不用规则伪造摘要。空 JSON
对象的重试会重新提交完整图文任务和字段要求。格式不完整、图片引用不合法或截断的
响应不会当作成功摘要。普通 HTTP 400 和 `data_inspection_failed` 只记录当前 POI
失败、不重试该请求，继续其他 POI；鉴权、余额、模型访问等全局错误仍会停止派发，
打印停止原因并保存正在处理的结果。具体分类参见
[百炼错误码](https://help.aliyun.com/zh/model-studio/error-code)。
可通过 `--workers`、`--request-interval`、`--timeout`、`--retries` 调整。
失败退出码为 1，中断为 130，参数错误为 2。

错误文件的 `attempts` 会记录脱敏并限长的 `provider_code`、`provider_message`、
`request_id`，以及字段缺失和图片引用问题的 `validation_details`。不保存完整
HTTP 错误正文、原始失败回答、API Key 或 Base64。2026-09-14 起，单张图片解码
失败会记录在 `images[].excluded_reason` 和 `unreadable_images`，保留原文件，
并使用同组可读图片生成摘要；`images_used` 和引用编号只计实际上传的图片。
若所有图片均不可读，记录 `no_decodable_images` 错误，不调用 API、不伪造摘要。
原始输入指纹仍覆盖全部文件；修复原图后会自动使缓存失效。

2026-09-14 修复了“任意 HTTP 400 都终止整批”的问题。原运行
`20260913T145359_dbd7b816` 的停止点为 `4d53a2c5a9378eec634eb2c4`，
少量原样重试确认服务端返回 `data_inspection_failed`。修复后的两 POI 验证中，
该 POI 记录失败后继续执行，`4d585ef592326ea8954d65c0` 已成功。
此改动没有改变初始提示词和图片预处理配置，原成功缓存继续有效。直接用相同命令
续跑即可，不要加 `--force`。内容检查拒绝的输入保持原样记录，不自动改变图片来规避检查。

用量统计包含收到用量信息的重试响应；网络超时等请求可能已在服务端计费却
未收到用量，记录为缺失，不能把本地统计当作最终账单。没有用量的有效摘要
可以保存，但运行报告会显式统计缺失。命中缓存不会再次调用 API。

## 证据边界

提示词仅使用图片，未读取评论、未来访问目标、用户历史或类别标签。
图片可能不相关、过时、分辨率低，模型生成的视觉描述仍需抽查，结构校验不等于
事实正确性。不会把图片推断出的活动当作已验证服务，也不推断实时营业情况、
价格、Wi-Fi、评分或用户偏好。这里生成视觉摘要，不宣称推荐指标已有改进。
