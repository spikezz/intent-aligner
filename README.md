# intent-aligner（意图对齐器）

意图对齐器由三部分组成：轨迹提取器、可视化，以及后续待定的部分。

轨迹提取器从一段对话的时间线开头开始，依次读取用户说的话、AI 对用户说的话、AI 的工具调用和环境返回的结果，提取其中的意图、观测以及它们之间的关系。轨迹提取器已经实现的部分是：把对话按时间顺序切成处理单元、第一级压缩工具操作、把每次发言（用户消息或 AI 对用户说的一段话）按意图切成段落，以及主循环，即逐个单元判断它属于哪个 topic、把发言的意图对照 topic 里同一方（用户或 AI）的已有意图、把工具调用的底层意图归入意图聚合并在聚合内去重、建立意图之间的关系，并判断每个观测达成或阻碍了哪些意图（见下面“运行”的第 3 到 5 步）。`playground/` 里是手工完成的示范提取，每个 topic 一个文件 `playground/topics/<topic 编号>.yaml`，所有 topic 列在 `playground/topics.yaml` 里。

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
# 4. 把工具调用和工具结果并行压缩成第一级概括，输出边跑边追加，中断后重跑会从断点继续
OUT=corpus/results/claude-5a1f0c1a-qwen38
mkdir -p $OUT/extract
$PY -m intent_aligner.compress corpus/merged/claude-5a1f0c1a.jsonl \
    --backend hive --model deployment/pc-qwen38-np6 --workers 6 --out $OUT/digests.jsonl

# 5. 指针按时间顺序逐个走过单元，每一步都带着此时全部 topic 的意图图谱：发言一段一段切分，判断单元属于哪个 topic，
#    对照已有意图，把工具调用的底层意图归入聚合并在聚合内去重，建立意图之间的关系，判断观测的影响。
#    全程串行，输出边跑边追加，中断后重跑会从断点继续。
$PY -m intent_aligner.extract corpus/merged/claude-5a1f0c1a.jsonl \
    --digests $OUT/digests.jsonl --segments $OUT/segments.jsonl \
    --backend hive --model deployment/pc-qwen38-np6 --out $OUT/extract
```

第 4 步的一次压缩如果思考达到了上限（`llm.HIVE_THINKING_BUDGET`），就用同一个提示词重新问，因为模型看不到上一次的思考，重新采样就是重试。一次压缩最多问 `compress.MAX_ATTEMPTS` 次，次数和校验不通过的重问共用。如果每一次都达到上限，就保留最后一个能通过校验的回答，并在这条压缩结果里写上 `review_reason`。如果每一次都达到上限、而且没有一个回答能通过校验，这次压缩就和其他失败一样报错，重跑同一条命令时会再试一次。第 5 步照常使用带 `review_reason` 的压缩结果，并把原因记进这个单元的 `capped_reviews`。

第 5 步的每一次调用（包括切分和放不下时的概括）也一样：思考达到上限就用同一个提示词重新问，一共 3 次仍然达到上限时，采用最后一个能通过校验的回答，把原因写进这个单元的 `capped_reviews` 和 `run.log`（以 `capped in` 开头），然后继续往下跑。如果 3 次顶满的回答都通过不了校验，就按回答不合格处理：用完重答次数仍然不通过时，这个单元标记 `needs_review`。

第 3 步输出的单元有 `user_message`（用户消息全文，附带图片路径）、`ai_text`（AI 对用户说的一段话）、`user_signal`、`tool_call` 和 `observation`。用户消息和 AI 说的话都算发言，用同样的方式处理。

切分时模型除了这次发言，还会看到全部 topic 的意图图谱和它之前 N 轮对话。一轮是一次发言，加上它之后、下一次发言之前的全部内容。这些对话里，工具操作显示第一级压缩的结果，放不下时最旧的一批工具操作会换成一段概括，所以第 5 步之前必须先跑完第 4 步。

N 由第 5 步的 `--window-rounds` 设定，默认值是 2。回放从 `run.sh` 里的这条命令读这个值，命令里没有写时按默认值显示。

切分时，模型重试几次仍然定位不到段落结尾，或者连一轮对话都放不进窗口时，剩下的文本成为一段 `needs_review: true`、意图为空的段落，原因打印在标准错误里。

第 5 步在 `--out` 目录下写出 `assignments.jsonl`（每个单元一行：单元的位置和它的 topic 编号，判断不出时为 null 并标记 `needs_review`）、`topics.yaml`、每个 topic 一个 `topics/<topic 编号>.yaml`（发言和用户信号的意图节点、工具调用的意图聚合节点、聚合的观测节点、其他观测节点、边、成对存放的工具调用和结果，以及段落；意图节点的 `actor` 标明是用户的还是 AI 的）、topic 判断不出的单元所在的 `topics/unassigned.yaml`，记录新 topic、整次没有意图的发言、`needs_review` 单元和采用了思考顶满回答的单元的 `run.log`，以及 `tool_compressions.jsonl`：模型看到的内容放不下时，最旧的一批工具操作会压缩成一段话，较新的保持逐条，每次压缩的结果记在这里。中断后重跑会从下一个单元继续。

第 1 步完成后，`corpus/index.json` 会列出每个合并后的对话，包括成员 session、记录条数、丢弃的 fork 复制品数量，以及损坏的行。

第 2 步写出的是原始记录，只改成了缩进排版，字段内容不变。

### 结果目录和重跑

第 4、5 步的结果按“对话 + 模型”放在同一个目录 `corpus/results/<platform>-<seed>-<模型>/` 下：`digests.jsonl`、`segments.jsonl` 和 `extract/`，也就是上面命令里的 `$OUT`。

第 4、5 步重跑时从已经写出的结果继续，只补缺少的部分，所以结果齐全时重跑不调用模型。续跑不检查模型是否和之前相同，所以换模型时要换一个目录，否则同一份结果里会混着两个模型的回答。它发现对话或切分结果和 `assignments.jsonl` 对不上时会报错停止，这时也要换一个 `--out` 目录。要完全重跑，先把这个目录里的 `digests.jsonl`、`segments.jsonl` 和 `extract/` 移走。只想重跑主循环时，可以把另一个目录的 `digests.jsonl` 复制到新目录，这时要连同同一目录的 `calls.jsonl` 一起复制，因为回放靠它找到第一级压缩的每次调用；只复制 `digests.jsonl` 时，回放页面上的第一级压缩点不开。

第 5 步在 `extract/` 下写一份 `calls.jsonl`，每次模型调用一行：调用的是哪一种（`call`、`segment`、`topic` 等）、为哪个任务调用（`tool:<tool_use_id>`、`round:<轮的 uuid>`、`message:<发言 uuid>`、`unit:<单元序号>`）、开始和结束时间、对话记录的路径，失败时还有错误信息。

### 回放一次运行

```bash
$PY -m intent_aligner.replay corpus/results/claude-5a1f0c1a-sol-rewrite
```

`intent_aligner.replay` 读结果目录的 `run.sh`，找到对话、模型和三步写出的文件，再把这次运行的每一次模型调用连同 `llm-home/` 里的原始请求和回答写成一个自包含的 HTML 页面，用浏览器直接打开。页面默认写到结果目录里的 `replay.html`，`--out` 可以改写到别处。新运行的 `run.sh` 照 `claude-87eeb4bd-qwen-final/run.sh` 在 compress 之前写两行：一行在后台用 `--watch 30` 启动回放，另一行用 `trap` 在脚本退出时停掉它并最后生成一次页面，这样运行一开始就有页面，不需要另外手动启动。运行还没结束时加 `--watch 30`，工具会每 30 秒用已有的数据重新生成页面，页面也每 30 秒重新载入，停在你正在看的那一帧，图谱在屏幕上的位置不变，展开和收起的栏目、各个栏目的滚动位置、图谱区域的高度以及图谱的缩放和平移都保持不变；播放时等播放停下才载入，按 Ctrl-C 结束。页面按时间顺序逐帧回放每次调用，播放时每帧分四拍：请求发出、端点处理、回答返回、代码校验。没有调用模型的抽取单元也占一帧，页面会说明代码为什么直接处理了它。

每一帧显示下面这些内容：

- 流程条标出当前阶段，主循环标到具体的调用（T、A、B、G、L、C、D）；点一个阶段或字母跳到下一次这一步的调用。
- 当前 topic 的图谱，放在当前任务的原文和三栏的上面：按 `assignments.jsonl` 逐单元重建。每一帧都显示图谱：单元还没写完时显示它开始前的图谱；调用 T 还不知道 topic、或者这一帧在切分和压缩时，显示最近一个单元所在 topic 到这一帧为止的图谱。图谱有改动时重新排版，没有改动时保持原来的排版。有边的节点放在虚线大框里，一条边都没有的节点放在左边单独的“没有边的节点”框里。本单元新增的节点和边在单元最后一次调用之前画成虚线，这次调用 header 里的候选加粗；用户的意图和 AI 的意图用不同的边框颜色，节点上也写明是谁的。已完成的意图和聚合整块涂满边框颜色、文字变成白色，右上角有绿底白字的“已完成”标记；待完成的在节点第一行写着“待完成”。聚合的观测和它所属的聚合之间画一条红色点线 `validates`，它不是存储里的边，而是观测节点的 `aggregate` 字段。点击节点或边显示全文，并列出改动过它的单元，点单元跳到它提交记录的那一帧；模型回答和写入记录里加引号的节点编号也可以点，点了在图谱里选中那个节点；节点可以拖动，页面会记住拖动后的位置，滚轮缩放，拖动空白处平移。双击一个节点，图谱以它为中心重排：它单独在最左边一列并保持选中，直接相连的节点排在右边第一列，它们的邻居再往右一列，依此类推；同一列节点之间的边向右弯出去绕过中间的节点，所以没有任何边会压到中心节点和它的连线。页面记住这个中心节点，“重新自动排版”恢复分层排版。
- 当前任务的原文：抽取单元的全文和它所属的 topic，或者发言里正在切分的那一段。
- 发给端点的请求：请求行里的设置、system 和提示词。提示词里源码写死的文字和随任务变化的文字用不同底色区分，重问时附上的反馈另外标出来。帧标题下面写明窗口实际放进了哪几轮，点一轮跳到主循环处理那一轮的第一个单元；旁边可以展开上下文窗口的规则，这段说明由页面从源码常量和 docstring 生成，不发给模型。
- 模型的回答、推理、用量和 schema。
- 代码对这个调用的校验规则和重问时追加的反馈，这些都是生成页面时从源码里读出来的；还有这次调用是第几次尝试（可以跳到同一个问题的上一次尝试和重问的下一次尝试）、被接受还是被驳回，以及写入的文件、needs_review 和 run.log 的行。
- token 统计，分两个标签页：思考 token 和完整上下文 token（提示词 + 思考 + 回答）。一个问题重新问了以后得到被接受的回答时，统计只算被接受的那一次，之前思考顶满、被驳回或出错的那几次不计入，页面写明有几次没有计入；重新问了仍然没有被接受的回答时，每一次都计入。每页一张图按 header 里图谱的节点数加边数排开，一张图从小到大排列（完整上下文的柱子分三段），思考 token 页的红色虚线是请求里设的思考上限，运行中途换过上限时只画最高的那个。每次调用是否顶满按它自己请求里的上限判断，判定线和 `llm.py` 相同（上限减去 `llm.HIVE_CAPPED_MARGIN`），悬浮提示显示这次调用自己的上限；表格里“思考顶到自己请求的上限”一列只数计入统计的调用。图上可以按调用种类筛选，点一个点或一根柱子跳到那次调用；柱状图可以横向缩放：+、−、Home 按钮，Ctrl+滚轮以光标处为中心缩放，滚轮或按住拖动左右平移。

依赖图按轮次画出全部任务。被标成 needs_review 的抽取单元里，没有发给模型的步骤用红色字母显示：`G✗` 表示这一步在路线上但没有发出去，`L?` 表示这一步要不要走取决于没有拿到的模型回答。第一层压缩的每个工具画成一对方块（压调用、压结果），结果分段压缩时标出段数；第一层压缩和主循环之间画一条屏障，写明 compress 命令结束后才运行 extract；同一轮发言切出的段落连成一条链；抽取单元画成一条链，每个单元标出它的调用序列，并高亮当前调用。页面顶部的表按阶段统计真实的调用次数、重试、驳回和失败，以及 `needs_review` 的任务数；点表里的数字跳到下一个符合条件的帧。`needs_review` 的任务在依赖图和时间条上用单独的颜色标出，这个任务的每一帧顶部都写明原因。

有 `calls.jsonl` 的运行直接按它找到每次调用；还没有写出任何结果的新运行被当成零次调用，不会退回到下面的按内容对应，所以不需要 `--since`。运行用 `--digests` 或 `--segments` 读另一个结果目录的文件时，回放也从那个目录的 `calls.jsonl`（切分是 `extract/calls.jsonl` 里的 segment 调用）取出对应的调用记录，并标明它们来自哪个目录。同一个任务用同一个基础提示词后来又从头问了一次时（例如换成更高的思考上限单独重跑一个调用，并把这次调用的记录追加到 `calls.jsonl`），回放采用最后一次的回答，前一次标成“后来同一个问题又从头问了一次”；这次重跑的调用排在它所属单元原有调用的后面，不按发出的时间排到最后。更早的运行没有这个文件，回放会在 `llm-home/` 里按模型、system prompt 和提示词里引用的对话原文把调用对应到任务上，并且只取这次运行最后一次写结果之前、中间没有超过 30 分钟停顿的那一段时间，因为同一个对话更早的运行也会留下一模一样的提示词。这段时间里同一个模型的调用如果对不上任何任务（发出它们的代码后来改过），页面会把这些记录列出来，不会猜着填进回放。另一个结果目录跑过同一个对话和同一个模型时，必须用 `--since` 和 `--until` 给出这次运行的时间范围。

### 调用 LLM

轨迹提取器可以通过 Claude Code、Codex 或 pi 的命令行调用模型，也可以直接通过 HTTP 调用 Hive Frame 网关上的本地模型。`extract` 的 `--backend` 只接受 `claude`、`codex` 和 `hive`，`pi` 只能在下面的 `intent_aligner.llm` 和手动测试工具里使用。每次调用都是独立的一问一答，prompt 从标准输入读入，回复打印到标准输出，对话记录的路径打印到标准错误。

```bash
echo "调查清楚 somnia-farm 项目的使用方法" | $PY -m intent_aligner.llm \
    --backend claude --model opus --effort xhigh \
    --system system.txt --schema schema.json
```

- `--backend` 可以是 `claude`、`codex`、`pi` 或 `hive`。`--model` 按对应命令行的写法给出，pi 的写法是 `provider/model-id`，hive 的写法是网关 `/v1/models` 列出的 `deployment/<部署名>`。
- `hive` 不经过任何 agent 命令行，发给网关的请求里只有系统提示和用户 prompt。网关地址默认是 `http://127.0.0.1:8790`，可以用环境变量 `HIVE_FRAME_URL` 改。思考参数按网关为这个部署声明的模型协议设置：Qwen 模板的模型打开思考，所有 effort 档位都用同一个思考预算 `llm.HIVE_THINKING_BUDGET`；单槽 `window.SLOT_TOKENS` 扣掉这个预算以后，剩下的由 prompt 和答案共用（`window.PROMPT_BUDGET`）。给了 `--schema` 时，schema 除了作为 `response_format` 约束输出，还以 `llm.HIVE_SCHEMA_INTRO` 开头写进系统提示的末尾，因为 `response_format` 只约束生成的 token，模型思考时读不到它。
- `--effort` 可以是 `low`、`medium`、`high` 或 `xhigh`，默认是 `xhigh`。
- `--stream` 和 `--no-stream` 决定 `hive` 用流式还是整块接收回复，默认是流式，对其他后端没有影响。`--no-stream` 的请求如果超过网关 1200 秒的读取超时，这个槽位就会一直被占着，直到部署重启。`extract` 也有这两个选项。
- `--schema` 是可选的 JSON schema，给出时回复就是满足它的 JSON。pi 不支持这个参数。
- Claude 调用显式启用 `--thinking-display summarized`，将可读思考摘要保存在原生 JSONL 中。回放读取其中的 `thinking`、普通回答或 `StructuredOutput`，以及 prompt snapshot 中的系统提示和 schema；空白或被省略的旧 thinking 无法补回。这里保存的是 Claude 返回的摘要，不是完整的原始推理。
- 调用产生的对话记录只写进 `--home` 指定的目录，默认是工具根目录下的 `llm-home/`。对话记录不会出现在日常使用的 `~/.claude/projects`、`~/.codex/sessions` 或 `~/.pi/agent/sessions` 里。

### 手动测试端点

`scripts/llm-chat.sh` 是一个交互式的小工具，用来手动给四个端点发消息。每条消息都是一次独立的调用，不会带上之前的消息。

```bash
PY=$PY scripts/llm-chat.sh                     # 默认从 claude 开始，也可以写 codex、pi 或 hive
PY=$PY scripts/llm-chat.sh --home /some/dir pi   # 对话记录写进 /some/dir
```

输入一行文字回车就会发出去，回复和对话记录路径会打印出来。以 `/` 开头的是命令：`/backend` 切换端点（模型同时换成这个端点的默认模型），`/model` 换模型，`/effort` 换思考强度，`/system` 换系统提示，`/show` 显示当前设置，`/help` 列出全部命令，`/quit` 退出。

## 测试

```bash
$PY -m pytest -q                            # 离线测试
INTENT_ALIGNER_LIVE=1 $PY -m pytest -q -k live   # 真实调用四个后端，会消耗 token；hive 需要网关上正在运行 deployment/pc-qwen38-np6
```

没有构建语料时，语料相关的测试会跳过；没有设置 `INTENT_ALIGNER_LIVE=1` 时，真实调用的测试会跳过。CI 只运行离线测试。

## 许可证

MIT，见 `LICENSE`。
