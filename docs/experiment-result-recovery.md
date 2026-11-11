# 实验结果保存与恢复

`experiment_results.record_reply()` 在统计用量前，将回答、转录路径和实验单元写入结果文件旁的 `*.receipts.jsonl`，加文件锁并 fsync。转录解析或 token 统计失败只记 `metadata_error`，已经保存的回答仍可恢复。

停止对应结果文件的收集进程后，可以离线恢复与刷新：

```python
from pathlib import Path
from intent_aligner.experiment_results import refresh_results

refresh_results(Path("results-direct-claude-opus-5-5.jsonl"))
```

它保留正式结果的回答及顺序，从 receipts 补入缺失单元，再从原始转录刷新用量。它不会调用任何模型；同一成功的 `(unit, run)` 不重复加入。原生 Claude JSONL 保持原样。

Claude 的 thinking 取 `output_tokens_details.thinking_tokens`；其余输出取 `output_tokens - thinking_tokens`，包含文本和结构化工具调用，字段名仍为 `answer`。输入计数包含 `input_tokens`、`cache_creation_input_tokens`、`cache_read_input_tokens`。缺少拆分字段时 tokens 保持未知，原始 usage 仍保留。

仓库里的回放实验脚本 [exp_direct.py](../scripts/exp_direct.py) 和 [exp_unverified.py](../scripts/exp_unverified.py) 按这个方式工作：每次调用的回答先经 `record_reply()` 写进 receipts，启动时先调用 `refresh_results()` 把上次中断时只写进 receipts 的回答补回结果文件，所以中断后重新运行只补缺少的 `(unit, run)`。
