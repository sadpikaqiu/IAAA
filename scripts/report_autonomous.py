"""Render existing agent experiment summaries without making model requests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def render_report(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    progress_path = directory / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    lines = ["# 自主 Agent 实验报告", "", f"状态：`{progress.get('status', 'unknown')}`；阶段：`{progress.get('stage', 'unknown')}`。",
             "", f"协议 SHA256：`{manifest.get('protocol_sha256', 'unavailable')}`。", "",
             "开发样本、独立验证和原全量测试分别报告，不能混合指标。外部图片/评论为观察日期未知的静态快照。", "",
             "A=fixed，B=fixed_llm_rank，C=fixed_schedule，D=autonomous；text 仅使用轨迹/结构化信息，both 加入图片摘要和评论。", "",
             "旧全量基线复用历史结果；TKY 图文唯一异常会话独立补跑，原文件保留。C/D 共享预算上限，实际耗时可能不同。", ""]
    for path in sorted((directory / "summaries").glob("*.json")):
        if "primary_comparisons" in path.name:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "arms" not in summary:
            continue
        lines += [f"## {path.stem}", "", f"已处理 {summary['n']}/{summary['expected_n']} 会话；质量通过：{summary['quality']['valid']}。", ""]
        if summary["quality"]["errors"]:
            lines += ["质量错误：" + ", ".join(summary["quality"]["errors"]) + "。以下逐组数值仅诊断已生成结果，不是完整配对指标。", ""]
        lines += ["| 配置 | n | Hit@1 | Hit@10 | NDCG@10 | 候选召回 | 原始召回 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, data in summary["arms"].items():
            m = data["overall"]
            lines.append(f"| {arm} | {data['n']} | {m['Hit@1']:.4f} | {m['Hit@10']:.4f} | {m['NDCG@10']:.4f} | {m['CandidateRecall']:.4f} | {m['RawCandidateRecall']:.4f} |")
        lines += ["", "| 配置 | 模型请求 | tokens | 会话耗时 P50/P95 (秒) |", "|---|---:|---:|---:|"]
        for arm, data in summary["arms"].items():
            cost = data["cost"]
            if cost.get("granularity") == "historical_full_run_aggregate":
                lines.append(f"| {arm} | 历史汇总 | {cost['full_run_total_tokens']}（原全量） | 未记录逐会话耗时 |")
            else:
                lines.append(f"| {arm} | {cost['requests']} | {cost['total_tokens']} | {cost['elapsed_p50']:.1f} / {cost['elapsed_p95']:.1f} |")
        if summary.get("timestamp_ties"):
            lines += ["", f"包含 {len(summary['timestamp_ties'])} 个同时间戳上下文会话；另见 strict_time_sensitivity 报告。"]
        lines += ["", f"逐组分层、配对区间、错误归因和用量：[JSON]({path.relative_to(directory).as_posix()})。", ""]
    for phase in ("validation", "full"):
        path = directory / "summaries" / f"{phase}_primary_comparisons.json"
        if path.exists():
            summaries = json.loads(path.read_text(encoding="utf-8"))
            lines += [f"## {phase} 预设主要比较", "", "| 城市/比较 | Hit@10 差值 | 用户聚类 95% CI | Holm p |",
                      "|---|---:|---|---:|"]
            for city, summary in summaries.items():
                for name, item in summary["contrasts"].items():
                    if "hit10_holm_p" not in item:
                        continue
                    lo, hi = item["user_cluster_bootstrap_95ci"]["Hit@10"]
                    lines.append(f"| {city}/{name} | {item['delta']['Hit@10']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {item['hit10_holm_p']:.4f} |")
            lines.append("")
    lines += ["## 解释边界", "", "- 候选召回提升需结合候选数量与检索成本解读。",
              "- 引用存在性由程序校验；引用是否充分支持语义判断还需检查案例。",
              "- B 的独立运行成本包含共享的 A 意图调用；physical_request_accounting 对真实请求去重。",
              "- 历史基线没有逐会话时间/token 明细，不用均分或零值冒充实测值。",
              "- 不根据测试准确率自动修改提示或选择样本。", ""]
    path = directory / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    print(render_report(parser.parse_args().results))
