# 运行结果

## `claude-5a1f0c1a-sol-l1/`

对话 `corpus/merged/claude-5a1f0c1a.jsonl` 按现行规则的运行，模型是 Codex 的 gpt-6.1-sol，思考强度 xhigh，窗口 `--window-rounds 2`。第一级压缩已经完成；主循环跑了一半就停了，那时 header 还是缩进排版，聚合观测还是追加式的写法。

| 文件 | 来自哪一步 | 内容 |
|---|---|---|
| `digests.jsonl` | `intent_aligner.compress` | 137 个工具调用和 137 个结果的第一级压缩，换模型以外的重跑都可以直接复用 |
| `segments.jsonl` | `intent_aligner.extract` 的主循环 | 主循环停下之前切好的发言段落 |
| `extract/` | `intent_aligner.extract` | 主循环停下之前的单元归属、topic 存储、压缩记录和 `calls.jsonl` |
| `run.sh`、`run.out`、`*.log` | | 运行命令和日志 |
| `replay.html` | `intent_aligner.replay` | 这次运行的回放页面 |

## `claude-5a1f0c1a-sol-rewrite/`

同一个对话、同一个模型和窗口，按紧凑 header 和每个结果重写一次聚合观测的规则跑完了主循环。第一级压缩（`digests.jsonl` 和它的 `calls.jsonl`）是从 `claude-5a1f0c1a-sol-l1/` 复制过来的，所以两次运行的主循环读的是同一份压缩结果。目录里的文件和上面相同。`tests/test_replay.py` 用这个目录的真实调用测试回放工具。

## `claude-87eeb4bd-qwen/`

对话 `corpus/merged/claude-87eeb4bd.jsonl`，模型是 Hive 网关上的 `deployment/pc-qwen38-np8`（Qwen 3.8 27B），思考强度 xhigh，窗口 `--window-rounds 2`，规则和 `claude-5a1f0c1a-sol-rewrite/` 相同。这次运行用来观察思考 token 随 header 里图谱变大怎样增长。目录里的文件和上面相同。

## 回放

用法见项目根目录 `README.md` 的“回放一次运行”：

```bash
$PY -m intent_aligner.replay corpus/results/claude-5a1f0c1a-sol-l1 --out replay.html
```

## 重跑

```bash
corpus/results/claude-5a1f0c1a-sol-l1/run.sh
```

`run.sh` 先做第一级压缩，再跑主循环。续跑和完全重跑的规则见项目根目录的 `README.md`（“结果目录和重跑”）。
