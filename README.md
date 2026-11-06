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

### 调用 LLM

轨迹提取器通过 Claude Code、Codex 或 pi 的命令行调用模型。每次调用都是独立的一问一答，prompt 从标准输入读入，回复打印到标准输出，对话记录的路径打印到标准错误。

```bash
echo "调查清楚 somnia-farm 项目的使用方法" | $PY -m intent_aligner.llm \
    --backend claude --model opus --effort xhigh \
    --system system.txt --schema schema.json
```

- `--backend` 可以是 `claude`、`codex` 或 `pi`。`--model` 按对应命令行的写法给出，pi 的写法是 `provider/model-id`。
- `--effort` 可以是 `low`、`medium`、`high` 或 `xhigh`，默认是 `xhigh`。
- `--schema` 是可选的 JSON schema，给出时回复就是满足它的 JSON。pi 不支持这个参数。
- 调用产生的对话记录只写进 `llm-home/`，不会出现在日常使用的 `~/.claude/projects`、`~/.codex/sessions` 或 `~/.pi/agent/sessions` 里。

### 手动测试端点

`scripts/llm-chat.sh` 是一个交互式的小工具，用来手动给三个端点发消息。每条消息都是一次独立的调用，不会带上之前的消息。

```bash
PY=$PY scripts/llm-chat.sh            # 默认从 claude 开始，也可以写 codex 或 pi
```

输入一行文字回车就会发出去，回复和对话记录路径会打印出来。以 `/` 开头的是命令：`/backend` 切换端点（模型同时换成这个端点的默认模型），`/model` 换模型，`/effort` 换思考强度，`/system` 换系统提示，`/show` 显示当前设置，`/help` 列出全部命令，`/quit` 退出。

## 测试

```bash
$PY -m pytest -q                            # 离线测试
INTENT_ALIGNER_LIVE=1 $PY -m pytest -q -k live   # 真实调用三个后端，会消耗 token
```

没有构建语料时，语料相关的测试会跳过；没有设置 `INTENT_ALIGNER_LIVE=1` 时，真实调用的测试会跳过。CI 只运行离线测试。

## 许可证

MIT，见 `LICENSE`。
