# 运行结果

## `claude-5a1f0c1a-sol-rewrite/`

对话 `corpus/merged/claude-5a1f0c1a.jsonl`，模型是 Codex 的 gpt-6.1-sol，思考强度 xhigh，窗口 `--window-rounds 2`，按紧凑 header 和每个结果重写一次聚合观测的规则跑完了主循环。`tests/test_replay.py` 用这个目录的真实调用测试回放工具。

| 文件 | 来自哪一步 | 内容 |
|---|---|---|
| `digests.jsonl` | `intent_aligner.compress` | 137 个工具调用和 137 个结果的第一级压缩，换模型以外的重跑都可以直接复用 |
| `segments.jsonl` | `intent_aligner.extract` 的主循环 | 切好的发言段落 |
| `extract/` | `intent_aligner.extract` | 单元归属、topic 存储、压缩记录和 `calls.jsonl` |
| `run.sh`、`run.out`、`*.log` | | 运行命令和日志 |
| `replay.html` | `intent_aligner.replay` | 这次运行的回放页面 |

## `claude-87eeb4bd-qwen/`

对话 `corpus/merged/claude-87eeb4bd.jsonl`，模型是 Hive 网关上的 `deployment/pc-qwen38-np8`（Qwen 3.8 27B），思考强度 xhigh，窗口 `--window-rounds 2`，规则和 `claude-5a1f0c1a-sol-rewrite/` 相同。这次运行用来观察思考 token 随 header 里图谱变大怎样增长。目录里的文件和上面相同。

## `claude-87eeb4bd-qwen-final/`

同一个对话，模型是 Hive 角色 `qwen38-pool`（`pc-qwen38-np6` 的 6 个槽位加 `art-qwen38-np2` 的 2 个槽位，第一级压缩用 8 个并发），思考强度 xhigh，窗口 `--window-rounds 2`，按现行提示词从空图谱重跑。第一级压缩、发言切分和主循环都在这一轮里做，所有调用都用同一个思考上限 `llm.HIVE_THINKING_BUDGET`。这一轮在 hive 请求把 schema 写进系统提示词之前开始，主循环跑到单元 17 时停掉，由 `claude-87eeb4bd-qwen-schema/` 接替；它的第一级压缩是 schema 修复前后对比的原始数据。

## `claude-87eeb4bd-qwen-schema/`

和 `claude-87eeb4bd-qwen-final/` 的模型、窗口和规则相同，hive 请求的系统提示词里带上了 schema。这次运行用来在 Qwen 上验证 `backups/intent-aligner-capped-thinking-rerun-2026-11-09/problem-table.md` 里的改动：调用 C 还会不会把思考顶满，serves 和 responds 怎样使用。

## `claude-87eeb4bd-sol/`

和 `claude-87eeb4bd-qwen-final/` 用同一套提示词跑同一个对话，模型换成 Codex 的 gpt-6.1-sol，思考强度 xhigh，作为对照。`run.sh` 会先续跑第一级压缩，压缩结果已经齐全，所以直接进入主循环。

## `claude-87eeb4bd-opus/`

同一个对话，模型是 Opus 5.5（`--backend claude --model claude-opus-5-5`），思考强度 xhigh，窗口 `--window-rounds 2`，按现行提示词从空图谱重跑。第一级压缩、发言切分和主循环都由 Opus 在这一轮里做，所以单元编号可能和 Qwen 的几轮不同，两轮对照时按发言的 uuid 对应单元。这次运行比 Qwen 快，用来先在 Opus 上验证问题表里的改动，特别是第 5 项：用户的同意记成 responds 之后，单元 27 和 59 还会不会把 serves 同时连到两层意图上。

## 回放

用法见项目根目录 `README.md` 的“回放一次运行”：

```bash
$PY -m intent_aligner.replay corpus/results/claude-5a1f0c1a-sol-rewrite
```

## 重跑

```bash
corpus/results/claude-87eeb4bd-qwen-schema/run.sh
```

`run.sh` 先做第一级压缩，再跑主循环。续跑和完全重跑的规则见项目根目录的 `README.md`（“结果目录和重跑”）。
