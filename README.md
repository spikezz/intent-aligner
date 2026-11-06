# intent-aligner（意图对齐器）

意图对齐器由三部分组成：轨迹提取器、可视化，以及后续待定的部分。

轨迹提取器从一段对话的时间线开头开始，依次读取用户说的话、AI 的工具调用和环境返回的结果，提取其中的意图、观测以及它们之间的关系。轨迹提取器目前还没有实现；`playground/` 里是手工完成的示范提取，每个 topic 一个文件 `playground/topics/<topic 编号>.yaml`，所有 topic 列在 `playground/topics.yaml` 里。

## 运行

运行环境是共享的 `/home/spikezz/.openclaw/.venv`。命令都在本目录下执行。

```bash
PY=/home/spikezz/.openclaw/.venv/bin/python

# 1. 构建语料：把 sessions.yaml 列出的对话拷贝到 corpus/raw/，
#    再把 fork 和重复的 session 合并成 corpus/merged/<platform>-<seed>.jsonl
$PY -m intent_aligner.corpus

# 2. 取出合并文件里第 N 条用户亲手发送的消息（从 0 开始计数）
$PY -m intent_aligner.turns corpus/merged/claude-5a1f0c1a.jsonl --index 0 --out playground/t
```

第 1 步完成后，`corpus/index.json` 会列出每个合并后的对话，包括成员 session、记录条数、丢弃的 fork 复制品数量，以及损坏的行。

第 2 步写出的是原始记录，只改成了缩进排版，字段内容不变。

## 许可证

MIT，见 `LICENSE`。
