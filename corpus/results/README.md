# 运行结果

`claude-87eeb4bd-sol/` 是对话 `corpus/merged/claude-87eeb4bd.jsonl` 按现行规则依次跑压缩、切分和主循环的结果，模型是 Codex 的 gpt-6.1-sol，思考强度 xhigh，窗口 `--window-rounds 2`，这次运行的主循环还没有结束。`claude-87eeb4bd-qwen-pool/` 只跑主循环，模型是 Hive 上的 `qwen38-pool`，思考强度 xhigh，窗口也是 2 轮；它直接读 `claude-87eeb4bd-sol/` 的压缩和切分结果，所以两次运行的差别只有主循环用的模型，这次运行还没有开始。

下面的内容说明 `claude-5a1f0c1a-*` 目录。

`claude-5a1f0c1a-sol/` 是对对话 `corpus/merged/claude-5a1f0c1a.jsonl` 依次跑压缩、切分和主循环的结果，模型是 Codex 的 gpt-6.1-sol，思考强度 xhigh。其中 `extract/` 是主循环只判断 topic 时的产物，没有意图和段落。`claude-5a1f0c1a-*` 的二级压缩和切分都只按用户消息划分，不包含 AI 发言；要得到现行规则的结果，需要从 `digests.jsonl` 里去掉二级压缩的行，再移走 `segments.jsonl` 和 `extract/` 后重跑。

| 文件 | 来自哪一步 | 内容 |
|---|---|---|
| `digests.jsonl` | `intent_aligner.compress` | 137 个工具调用和 137 个结果的一级压缩，7 段每轮的二级压缩；每个调用的意图带 evidence、explicit 和 note |
| `segments.jsonl` | `intent_aligner.segment` | 7 条用户消息切成的 8 个段落，每行一条消息；每个意图带 evidence、explicit 和 note |
| `extract/assignments.jsonl` | `intent_aligner.extract` | 298 个单元，每行一个单元和它的 topic |
| `extract/tool_compressions.jsonl` | `intent_aligner.extract` | 一轮太长时，这一轮工具记录的 3 次压缩 |
| `extract/topics.yaml`、`extract/topics/` | `intent_aligner.extract` | topic 列表和每个 topic 的存储文件；这次运行只判断了 topic，所以图谱和段落是空的 |
| `extract/run.log` | `intent_aligner.extract` | 新 topic 和 `needs_review` 单元的记录 |
| `*.log`、`extract/stdout.log` | 各步的运行日志 | |

`claude-5a1f0c1a-qwen/` 里的 `digests.jsonl` 和 `segments.jsonl` 由本地 Qwen（Hive 的 `deployment/pc-qwen38-np8`，思考强度 xhigh）生成。这个目录的 `extract/` 是一次没有跑完的主循环结果，同样只按用户消息划分。更早的主循环结果在下面两处，都只覆盖对话的一部分：

| 路径 | 内容 | 生成脚本 |
|---|---|---|
| `extract-before-lookahead-qwen51-sol70/` | 调用 G 还看不到后文时的前 121 个单元，前 51 个由 Qwen 跑，其余 70 个由 sol 跑 | `run-first-120.sh`、`run-rest-120-sol.sh` |
| `lookahead-slices/from-<K>/` | 调用 G 能看到后文时的验证切片：拷贝上面结果的前 K 行，再由 Qwen 往后跑 20 个单元，K 是 41、101、121 | `run-lookahead-slices.sh` |

`claude-5a1f0c1a-opus/segments.jsonl` 是 Claude 的 opus（思考强度 xhigh）只跑切分的结果，用的是旧格式的 sol 一级压缩（每个调用一句话意图）。`segment-intents-comparison.md` 是旧格式下 sol、opus、Qwen 三组切分的并排对比，Qwen 那一列已经被新的完整运行取代。

## 结果

sol：

- 压缩：137 个调用、137 个结果、7 段二级压缩，没有重复。137 个调用一共写了 234 个意图（68 个调用 1 个、45 个 2 个、20 个 3 个、4 个 4 个），其中明说 8 个、推断 226 个；这个对话的 Bash 调用都没有 `description` 字段，明说的意图都来自命令里的注释或脚本说明。
- 切分：7 条消息切成 8 段，没有 `needs_review`；每条消息的段落首尾相接覆盖整个正文。只有 c88f10d9 切成两段，其余各一段。
- topic 归属：298 个单元全部归到 T001“研究 somnia-farm 项目的使用方法，为制作《The Wind Remembers》音乐 MV 做准备。”，没有 `needs_review`。调用 T 调用了 161 次，每次都是第一次回答就通过校验；其余 137 个单元是工具结果，沿用调用的 topic。
- 工具记录压缩：一共 3 次，第一轮 1 次，倒数第二轮 971b4279 2 次，第二次从第一次结束的位置开始。

Qwen：

- 压缩：第一次运行有 1 个调用失败。那条 Bash 命令里有很多带引号的路径，Qwen 在 evidence 的 JSON 转义上反复计数，生成 18,518 个 token 用完单槽后以 `length` 截断（约 33 分钟）；重跑这一个调用后通过。137 个调用一共写了 238 个意图（65 / 50 / 15 / 7），明说 5 个、推断 233 个，拆分的细度和 sol 相近。
- 切分：7 条消息切成 9 段，没有 `needs_review`；c88f10d9 切成三段。
- topic 归属（只判断 topic 的那次运行，结果已经不在目录里）：299 个单元，296 个归到 T001，3 个归到 T002“记录 Claude Code 因 Opus 5.5 安全护栏拦截消息而返回 API 错误……”。T002 的三个单元都是对话末尾的观测：一条被拦截后合成的 AI 回复和之后的两条观测；sol 把它们归在 T001。没有 `needs_review`，调用 T 调用 162 次，每次第一次回答就通过。
- 工具记录压缩：3 次，和 sol 相同。

## 回放

每个结果目录都可以用 `intent_aligner.replay` 回放成一个本地 HTML 页面，用法见项目根目录 `README.md` 的“回放一次运行”：

```bash
$PY -m intent_aligner.replay corpus/results/claude-87eeb4bd-sol --out replay.html
```

## 重跑

```bash
corpus/results/claude-5a1f0c1a-sol/run.sh
```

每个目录的 `run.sh` 依次跑三步，写到这个目录的 `extract/`。续跑和完全重跑的规则见项目根目录的 `README.md`（“结果目录和重跑”）。
