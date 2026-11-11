# intent-aligner：Qwen 思考顶满的问题表（已解决的档案）

从 `problem-table.md` 移过来的已解决问题。没解决的问题留在 `problem-table.md`，编号两边通用。
移入时间：2026-11-11 00:18。

实验的原始数据都在 `/home/spikezz/.openclaw/backups/intent-aligner-capped-thinking-rerun-2026-11-09/` 下：

- `results-exp-prompts.jsonl`：base / unit / direct 三个变体，跑到 36/64 时停掉。
- `results-step.jsonl`：单元 30 和单元 64 用专用 C 提示词回放的结果。
- `results-serves.jsonl`：单元 27 和单元 53 用 serves 上一版措辞（“手段、前提，或者把 to 具体化”）回放的结果。
- `results-final.jsonl`：最终版提示词回放单元 30、59、64，以及切分提示词删掉“条件、”的实验。
- `results-responds.jsonl`：加 responds、同一次发言不连之后，回放单元 7–10 和 sol 一轮的单元 55、63、98。
- `results-direct.jsonl`：serves 改成“from 直接服务于 to”之后的回放，11 个单元各 3 次，已跑完（`exp_direct.py`）。
- `results-unverified-claude-opus-5-5.jsonl`：现行提示词在 Opus 5.5 上回放单元 30、64（专用提示词）和 58、59，每个单元 3 次，已跑完（`exp_unverified.py <目录> claude claude-opus-5-5`）。
- `results-direct-claude-opus-5-5.jsonl`：同样 33 个请求发给 Opus 5.5（claude 后端，xhigh）的对照，已跑完（`exp_direct.py <目录> claude claude-opus-5-5`）。已从原生转录离线补齐 33 条 token 统计：thinking 取 `output_tokens_details.thinking_tokens`，其余输出取 output_tokens 减 thinking；输入包含缓存读写。单元 30、64、58、59 的另外 12 条结果也已补齐。第一轮崩溃前完成的 7 条回答仍在原生转录中，已单独恢复到 `/media/spikezz/v/somnia-farm-audit/data/tasks/intent-aligner-claude-accounting-20261110/results-direct-claude-opus-5-5.recovered-first-batch.jsonl`，不混入这 33 条对照；没有重新调用模型。
- `exp-capped/`、`capped-run-2026-11-10/`：思考原文、提示词和回答。

## 已解决

| # | 状态 | 问题 | 出现在哪些单元 | 怎么处理的 | 验证 |
|---|---|---|---|---|---|
| 1 | **Opus 和 Qwen 都通过** | 已有的边要不要再写一遍 | 30、64 | 工具调用并入已有聚合时，C 改用专用提示词 `C_STEP_SYSTEM_PROMPT`，只问这一步新表现出来的关系。发言单元和新开聚合的工具调用仍用 `C_SYSTEM_PROMPT` | 部分验证。单元 30 和单元 64 各跑 3 次：思考 3,026 到 9,090 token，没有顶满；6 次回答都是空列表，对照原文判断是正确的。但这次回放用的是改动前的提示词，第 3、5、6 项的改动都还不在里面。**Opus 上已验证**：现行提示词（专用提示词、原措辞“产生于这个聚合之外”、serves 直接服务于、responds）在 Opus 上回放单元 30 和 64 各 3 次（`results-unverified-claude-opus-5-5.jsonl`），6 次回答都是空列表，和上面判断正确的答案一致。Qwen 上的顶满情况要看整轮重跑。**Qwen 上已验证**：同样的现行提示词在 `deployment/pc-qwen38-np6` 上回放单元 30 和 64 各 3 次（`results-unverified.jsonl`），6 次都没有顶满，思考 1,868 到 10,008 token。单元 30 的 3 次和单元 64 的 2 次是空列表，和 Opus 一致；单元 64 第 0 次写了 `O006 triggers G006`，这是第 6 项的问题，见第 6 项。思考最多的正是写错的那一次（10,008，离顶满的判定线 10,176 只差 168），所以单元 64 仍然贴近上限 |
| 2 | 不需要模型验证（代码去掉，有单元测试） | 同一个聚合内部要不要记 triggers | 多个单元 | Spike 定保持现状：聚合内部不记边，代码会把这种边从回答里去掉（`extract.py` 的 `_relate.parse`） | 不需要模型回放：代码去掉这种边，有单元测试覆盖 |
| 3 | **Opus 和 Qwen 都通过** | `<当前单元>` 指整次发言，还是 `<当前段落>` | 58、59 | 采用 unit 变体：`<当前单元>` 的说明换成"发言段落给出整次发言，用 `<当前段落>` 标出这一段；工具调用给出它的意图和操作对象的概括"。这句话在共用的 `_C_TASK` 里，两份 C 提示词都用 | 部分验证。单元 59 在最终版提示词下跑 2 次，1 次顶满（`results-final.jsonl`，当时 serves 还是上一版措辞）。发言单元顶满次数从 base 的 4/10 降到 1/10；单元 27 和单元 58 上给出的边和 base 相同。每个单元只跑了 2 到 3 次；单元 59 是因为这个问题列进来的，但实验停在它之前，一次都没跑到。**Opus 上已验证**：现行提示词回放单元 58、59 各 3 次（单元 59 的 candidates 去掉了同一次发言里的 I010），每条边的一端都是这个段落自己的意图，没有给同一次发言里的另一个段落连边，同一个单元的 3 次回答完全一致。单元 58 是 `O005 triggers I010`、`I010 serves I008`；单元 59 是 `O005 triggers I011`、`I011 serves I005`、`I011 serves I003`（这两条 serves 的问题见第 5 项）。**Qwen 上已验证**：同样回放在 `deployment/pc-qwen38-np6` 上各跑 3 次（`results-unverified.jsonl`），都没有顶满，思考 2,425 到 8,004 token。单元 58 的 3 次和 Opus 完全相同。单元 59 的 3 次都包含 Opus 的那三条边，第 2 次多了一条 `I011 serves I007`。6 次都没有给同一次发言里的另一个段落连边 |
| 4 | **Opus 和 Qwen 都通过** | 并入已有聚合的工具调用还要不要调用 C | 30、64 | 和第 1 项是同一件事：照 Spike 的方案，没有跳过 C，而是改用专用提示词 | 同第 1 项，Opus 和 Qwen 上都已验证 |
| 13 | Opus、Qwen、sol 上环都照样出现（Spike 定可接受） | 聚合和意图之间出现两个方向的 serves（环） | sol 上一轮整轮重跑：G003↔I011、G004↔I018/I019 | AI 在聚合 G003 中途宣布下一步（I011，跑 lint），宣布之后的工具调用又并入 G003，于是同时有 `I011 serves G003` 和 `G003 serves I011`。Spike 定：环的一边是聚合、一边是意图，聚合横跨了上下游，这种环可以接受，不改代码也不加规则 | Opus 对照回放（新措辞）里环照样出现：G003 并入的那一步 3 次都写 `G003 serves I011`（另加 `serves I012`），宣布 I011 的发言 3 次都写 `I011 serves G003`；G004 和 I019 同样两个方向都有，3/3。宣布 I011 的发言另外 3 次都写了 `O003 triggers I011`，也就是 G003 前半段的结果引出了 I011。**Qwen 回放（`results-direct.jsonl`）：** 宣布 I011 的发言 3 次都写 `I011 serves G003`，G003 并入的那一步 3 次都写 `G003 serves I011`（另加 `serves I012`）；I018、I019 各 3 次都写 `serves G004`，G004 那一步 3 次都写 `serves I018`、`serves I019`。这 18 次的思考都在 352 到 1,135 token，没有因为反方向的边犹豫或顶满。sol 整轮重跑里 5 个环都出现了。原来要看的：整轮重跑跑到单元 60–100 时：新措辞下环还出不出现；模型写这类边时有没有因为图谱里已有反方向的边而反复犹豫、顶满 |
| 14 | **Qwen 通过**（已实现，cda1b7e；回放和整轮都验证过） | hive 请求的 schema 模型看不到，targets 被截坏 | qwen-final 压缩：51 次压调用里 6 次 targets 是 `...`、`[` 或字面的 `targets`，另有 5 次只剩一个路径 | hive 只把 schema 放在 `response_format` 里，它只约束生成的 token，模型思考时读不到，于是按自己猜的格式（例如数组）规划 targets，输出时被语法约束截短。Spike 定把 schema 写进提示词：`llm.hive_request_body` 在系统提示末尾加上 `llm.HIVE_SCHEMA_INTRO` 和 schema 原文，回放页面的系统提示改为显示请求里实际发出的内容 | 单次重放（`results-compress-capped.jsonl`）：原样 3 次里 2 次 targets 坏，写进 schema 后 3 次都完整，思考 token 没有减少。51 次压调用全量重放（`exp_schema_in_prompt.py` → `results-schema-in-prompt.jsonl`，pc np6，各 1 次）：原来 11 次 targets 坏（6 次明显坏、5 次只剩一个路径，重放后这 5 次都列出了更多对象，说明它们也是被截短的），重放后 51 次全部完整；原来 1 次顶满，重放后 0 次；生成 token 中位数 1,981 → 1,348、最大 10,527 → 6,783。**整轮也通过**：qwen-schema 一轮用这版代码做完了全部压缩（11-10 22:22–22:46），51 个工具调用和 51 个结果，没有一次顶满、没有调用失败、没有明显坏掉的 targets；压调用的思考中位数 1,345、最大 6,840，压结果中位数 495、最大 1,379 |

## responds 的补验证

- responds 的含义按 Spike 的原话定为“from 回应对方说的话 to：一端是用户，另一端是 AI”，代码只接受一端是用户的意图、另一端是 AI 的意图。Spike 又定：同一次发言里的几个意图之间不连边。调用 C 的 candidates 不再列出同一次发言的意图，提示词里“另一端在 candidates 或 unit_intents 里”改成“另一端在 candidates 里”，代码把两端出自同一次发言的边从回答里去掉。`exp_responds.py` → `results-responds.jsonl` 回放了单元 7、8、9、10（单元 8、9 的录制 candidates 去掉了同一次发言的 I002、I003）和 sol 一轮的单元 55、63、98，每个单元 3 次，已跑完。Qwen 的 12 次都没有顶满，思考 1,081 到 8,922 token；每次都用 responds 连到对方的话：单元 7 是 I002→I001，单元 8 是 I003→I001，单元 9 是 I004→I001，单元 10 是 I005→I004（有 1 次另加 I005→I003）。单元 8 有 1 次多出 `G001 serves I003`。sol 的 9 次思考都在 285 到 757 token，回答 3 次完全一样：做完活之后的汇报 I010、I014、I017 仍然用 responds 连到 I008。I008 是用户同意在当前会话里依次排查 3070、选关键帧、推进镜头，这几条汇报是 AI 做完其中的事以后对用户说的话，所以它们符合“回应对方说的话”的字面意思；Spike 定：这样连不算错，保持现状。其他单元的录制 candidates 里哪些出自同一次发言没有逐个核对，交给整轮重跑 `claude-87eeb4bd-qwen-final` 和 `claude-87eeb4bd-sol`（从空图谱重新开始）。

## sol 整轮重跑（claude-87eeb4bd-sol）里和已解决项有关的结果

- **第 13 项的环在 sol 上照样出现，一共 5 个：** G002↔I009（单元 30、31）、G003↔I011 和 G003↔I012（单元 56、57、58）、G004↔I018 和 G004↔I019（单元 99、100、101）。形状和第 13 项描述的一样：AI 在聚合中途宣布下一步，之后的工具调用又并入这个聚合。Spike 已定这种环可以接受。
- **responds：** 25 条，分布在单元 7–13、16、55、63、98、104、111–121。单元 55、63、98 是“已处理”里 responds 那一条记录过的做完活之后的汇报连到 I008，Spike 已定不算错。
