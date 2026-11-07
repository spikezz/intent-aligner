# intent-aligner（意图对齐器）

意图对齐器由三部分组成：轨迹提取器、可视化，以及后续待定的部分。

轨迹提取器从一段对话的时间线开头开始，依次读取用户说的话、AI 的工具调用和环境返回的结果，提取其中的意图、观测以及它们之间的关系。轨迹提取器已经实现的部分是：把对话按时间顺序切成处理单元、两级压缩工具操作、把用户消息按意图切成段落，以及主循环的开头，即逐个单元判断它属于哪个 topic（见下面“运行”的第 3 到 6 步）；之后建节点和关系的步骤还没有实现。`playground/` 里是手工完成的示范提取，每个 topic 一个文件 `playground/topics/<topic 编号>.yaml`，所有 topic 列在 `playground/topics.yaml` 里。

## 运行

运行环境是共享的 `/home/spikezz/.openclaw/.venv`。命令都在本目录下执行。

```bash
PY=/home/spikezz/.openclaw/.venv/bin/python

# 1. 构建语料：把 sessions.yaml 列出的对话拷贝到 corpus/raw/，
#    再把 fork 和重复的 session 合并成 corpus/merged/<platform>-<seed>.jsonl
$PY -m intent_aligner.corpus

# 2. 取出合并文件里第 N 条用户亲手发送的消息（从 0 开始计数）
mkdir -p corpus/turns
$PY -m intent_aligner.turns corpus/merged/claude-5a1f0c1a.jsonl --index 0 --out corpus/turns/claude-5a1f0c1a-0.json

# 3. 把合并后的对话（Claude 或 Codex，按文件名前缀区分）按时间顺序切成处理单元，
#    每行一个单元，并打印各类单元的数量
mkdir -p corpus/units
$PY -m intent_aligner.units corpus/merged/claude-5a1f0c1a.jsonl --out corpus/units/claude-5a1f0c1a.jsonl
```

```bash
# 4. 把一个对话的工具操作分两级压缩。第一级写出每个工具调用的意图（每个意图附带原文摘录 evidence、
#    是明说还是推断 explicit 和推断依据 note）和它针对的具体对象，把它的
#    结果对照调用的意图和全文压成一句话（太长的结果按行切段，每段一句），一个调用和它的结果是一个任务，
#    任务之间互相独立，--workers 个并行。第二级把一轮里全部的意图、对象和结果压成一段概括。
#    第二级同时看开启这一轮的用户消息。输出边跑边追加，中断后重跑会从断点继续；--until 在这条用户消息之前停下。
OUT=corpus/results/claude-5a1f0c1a-qwen38
mkdir -p $OUT
$PY -m intent_aligner.compress corpus/merged/claude-5a1f0c1a.jsonl \
    --backend hive --model deployment/pc-qwen38-np8 \
    --out $OUT/digests.jsonl

# 5. 把对话里的每条用户消息按意图切成连续的段落，每行一条消息。每个意图附带原文摘录 evidence、
#    是明说还是推断（explicit）和推断依据 note。--workers 条消息并行，
#    输出边跑边追加，中断后重跑会从断点继续。第 4 步必须先对整个对话跑完（不加 --until）
$PY -m intent_aligner.segment corpus/merged/claude-5a1f0c1a.jsonl \
    --digests $OUT/digests.jsonl \
    --backend hive --model deployment/pc-qwen38-np8 --out $OUT/segments.jsonl

# 6. 指针按时间顺序逐个走过单元（用户消息换成第 5 步的段落），判断每个单元属于哪个 topic。
#    工具结果沿用它的调用的 topic，其余单元各调用一次模型。第 4、5 步都必须先对整个对话跑完
$PY -m intent_aligner.extract corpus/merged/claude-5a1f0c1a.jsonl \
    --digests $OUT/digests.jsonl --segments $OUT/segments.jsonl \
    --backend hive --model deployment/pc-qwen38-np8 --out $OUT/extract
```

第 3 步输出的单元有 `user_message`（用户消息全文，附带图片路径）、`user_signal`、`tool_call` 和 `observation`；另外 `ai_text` 是 AI 对用户说的话，只注入 input，不做处理。

第 5 步的模型除了这条消息，还会看到它之前两轮对话；两轮放不下时只看前一轮。这些对话里，每一轮的工具操作只显示第 4 步写出的那一段概括，所以这条消息之前的对话必须先压缩过。

第 5 步里，模型重试几次仍然定位不到段落结尾，或者连一轮对话都放不进窗口时，剩下的文本成为一段 `needs_review: true`、意图为空的段落，原因打印在标准错误里。

第 6 步在 `--out` 目录下写出 `assignments.jsonl`（每个单元一行：单元的位置和它的 topic 编号，判断不出时为 null 并标记 `needs_review`）、`topics.yaml`、每个 topic 一个 `topics/<topic 编号>.yaml`（图谱和段落暂时为空），记录新 topic 和 `needs_review` 单元的 `run.log`，以及 `tool_compressions.jsonl`：一轮太长、模型看到的内容放不下时，还没压缩过的工具操作会压缩成一段话，之前压缩出的段落保持不变，每次压缩的结果记在这里。中断后重跑会从下一个单元继续。

第 1 步完成后，`corpus/index.json` 会列出每个合并后的对话，包括成员 session、记录条数、丢弃的 fork 复制品数量，以及损坏的行。

第 2 步写出的是原始记录，只改成了缩进排版，字段内容不变。

### 调用 LLM

轨迹提取器可以通过 Claude Code、Codex 或 pi 的命令行调用模型，也可以直接通过 HTTP 调用 Hive Frame 网关上的本地模型。每次调用都是独立的一问一答，prompt 从标准输入读入，回复打印到标准输出，对话记录的路径打印到标准错误。

```bash
echo "调查清楚 somnia-farm 项目的使用方法" | $PY -m intent_aligner.llm \
    --backend claude --model opus --effort xhigh \
    --system system.txt --schema schema.json
```

- `--backend` 可以是 `claude`、`codex`、`pi` 或 `hive`。`--model` 按对应命令行的写法给出，pi 的写法是 `provider/model-id`，hive 的写法是网关 `/v1/models` 列出的 `deployment/<部署名>`。
- `hive` 不经过任何 agent 命令行，发给网关的请求里只有系统提示和用户 prompt。网关地址默认是 `http://127.0.0.1:8790`，可以用环境变量 `HIVE_FRAME_URL` 改。思考参数按网关为这个部署声明的模型协议设置：Qwen 模板的模型打开思考，`low` 给 2048 个思考 token 的预算，`medium`、`high`、`xhigh` 都给 9216 个，这样单槽 18,944 token 里还放得下 prompt 和答案。
- `--effort` 可以是 `low`、`medium`、`high` 或 `xhigh`，默认是 `xhigh`。
- `--stream` 和 `--no-stream` 决定 `hive` 用流式还是整块接收回复，默认是流式，对其他后端没有影响。`--no-stream` 的请求如果超过网关 1200 秒的读取超时，这个槽位就会一直被占着，直到部署重启。`compress` 和 `segment` 也有这两个选项。
- `--schema` 是可选的 JSON schema，给出时回复就是满足它的 JSON。pi 不支持这个参数。
- 调用产生的对话记录只写进 `--home` 指定的目录，默认是工具根目录下的 `llm-home/`。对话记录不会出现在日常使用的 `~/.claude/projects`、`~/.codex/sessions` 或 `~/.pi/agent/sessions` 里。

### 手动测试端点

`scripts/llm-chat.sh` 是一个交互式的小工具，用来手动给三个端点发消息。每条消息都是一次独立的调用，不会带上之前的消息。

```bash
PY=$PY scripts/llm-chat.sh                     # 默认从 claude 开始，也可以写 codex、pi 或 hive
PY=$PY scripts/llm-chat.sh --home /some/dir pi   # 对话记录写进 /some/dir
```

输入一行文字回车就会发出去，回复和对话记录路径会打印出来。以 `/` 开头的是命令：`/backend` 切换端点（模型同时换成这个端点的默认模型），`/model` 换模型，`/effort` 换思考强度，`/system` 换系统提示，`/show` 显示当前设置，`/help` 列出全部命令，`/quit` 退出。

## 测试

```bash
$PY -m pytest -q                            # 离线测试
INTENT_ALIGNER_LIVE=1 $PY -m pytest -q -k live   # 真实调用四个后端，会消耗 token；hive 需要网关上正在运行 deployment/pc-qwen38-np8
```

没有构建语料时，语料相关的测试会跳过；没有设置 `INTENT_ALIGNER_LIVE=1` 时，真实调用的测试会跳过。CI 只运行离线测试。

## 许可证

MIT，见 `LICENSE`。
